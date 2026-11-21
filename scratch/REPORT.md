# 近重复索引缓冲维护链路修复报告

## 1. 缺陷与根因

复现（`scratch/repro_baseline.py`，MinHash 长度 8、`params=(2,4)`、同一 `insertion_session`
先插全 1 再插全 2 的 hashvalues）在修复前的输出留存于 `scratch/baseline_output.txt`：

- Redis 路径第二次插入被接受（dict 路径报 `ValueError`）；关会话后两种查询都返回 `k`；
  `remove(k)` 后成员检查为 `False`，全 2 查询仍返回 `k`；键到签名存储里 `k` 的签名被
  `RPUSH` 追加成两份，而 `_remove` 的 `zip(keys[key], hashtables)` 只清理前 `b` 条，
  其余签名残留在桶里形成"幽灵命中"。
- 删除会话（`scratch/probe_deletion_baseline.py`）：`SREM` 只进了缓冲，`_remove` 里的
  空桶检查 `hashtable.get(H)` 读到的是刷新前状态，`remove(H)` 从不触发，桶统计
  （`get_counts`）留下零成员条目，全部删完后 `is_empty()` 仍为 `False`；同一会话里
  重复删除同一键被接受（应报"键不存在"）。

根因是一条逻辑变更涉及几份独立缓冲（键→签名存储 + 每 band 一个桶存储），它们各自在
容量满时自动写出，而**存在性判断只读已落地状态**，没有覆盖"已接受但未刷出"的操作：

1. `RedisListStorage.has_key` 只发 `HEXISTS`，看不到缓冲里的 `HSET`/`HDEL`，于是
   重复插入、重复删除都被放行；
2. 空桶确认在错误的时刻做（变更时读旧状态），且缓冲容量因 MRO 遮蔽从未生效（见下）；
3. 附带发现的既有问题：
   - `Storage.buffer_size` 在 MRO 上遮蔽 `RedisStorage.buffer_size`，会话的
     `buffer_size` 从不传播到 `RedisBuffer`（自动刷新阈值永远是 50000）；
   - `RedisListStorage.remove_val`/`RedisSetStorage.remove_val` 非缓冲分支用
     `hdel(name, redis_key)` 删了一个不存在的字段（应为 `key`），空桶条目永远删不掉；
   - `RedisListStorage.remove_val` 的 `lrem(redis_key, val)` 是 redis-py 2.x 的两参
     形式，在 redis-py ≥3 下必抛 `TypeError`（LSH 路径没用到它，故从未暴露）；
   - redis-py 在"建连失败"时保留命令栈、"发送失败"时丢弃，失败批次之后会被悄悄重放。

## 2. 修复设计

### 待处理键的身份放在哪里、怎样跟随实际刷新收回

存在性簿记放在**各存储实例**上（不是会话级集合），用缓冲执行钩子收回：

- `RedisBuffer` 增加 `on_execute` 回调；`execute()` 无论成功失败、自动还是显式触发，
  都在 `finally` 里调用。失败时先 `reset()` 丢弃命令栈（统一 redis-py 的不一致行为），
  保证"失败批次被丢弃、不会在之后被重放"。
- `RedisListStorage`（键→签名）持有 `_pending_inserts`/`_pending_removes` 两个集合，
  只在命令入队**之后**登记（入队可能触发自动刷新，刷新即清空）；`has_key` 先查
  `_pending_removes`（→False）再查 `_pending_inserts`（→True），否则回退 `HEXISTS`。
  已落地的键由 `HEXISTS` 覆盖，未刷出的键由簿记覆盖——**接纳结果与缓冲大小无关**，
  簿记大小以缓冲容量为上界，长会话不会越记越多。
- `RedisSetStorage`（桶）通过 `_note_*` 空实现把簿记从 `has_key` 上摘掉，**查询的
  中途可见性保持原样**；它另持 `_pending_srem_buckets`（被缓冲 SREM 触及的桶键）。

### 删除后的桶目录何时确认

推迟到**刷新时**确认：缓冲执行后，对该批次触及的桶做一次管道化 `SCARD`，为 0 的桶
`HDEL` 掉目录条目（`_confirm_empty_buckets`）。桶存储因此获得
`deferred_bucket_cleanup = True`，`MinHashLSH._remove` 在缓冲模式下对它跳过逐键的
`SMEMBERS` 空桶探测（dict 与非缓冲路径行为不变）。

### 会话生命周期

`_MinHashLSHSession` 统一两类会话：`close()` 幂等（重复 close 无变更），逐个存储尝试
刷新、把首个后端失败重抛给调用方（不当作成功关闭，已执行命令不回滚），会话随后视为
已关闭；已关闭会话的 `insert`/`remove`/`__enter__` 一律 `ValueError`。

### 被否决做法

- **会话级"已处理键"集合**：内存随会话长度增长，违反"簿记只随未完成的缓冲工作增长"。
- **每键强制刷管道**：把批处理退化成逐键写回（任务明确禁止）。
- **变更前遍历索引/桶目录**：O（索引大小） 每次变更（任务明确禁止）。
- **在 `get` 上叠加未刷出 SREM**：会让 `query` 在会话中途看到未落地删除，改变既有
  可见性语义；且要按桶记成员值，簿记更大。

## 3. 可行做法的代价比较

| 做法 | 额外往返 | 批处理 | 簿记空间 | 中途可见性 |
|---|---|---|---|---|
| A. `get` 叠加 SREM 簿记（变更时确认） | 每删除每 band 1 次 `SMEMBERS`（=既有开销） | 写仍批量 | 每桶成员值集合，≤缓冲容量 | 改变（删除提前可见） |
| B. 刷新时 `SCARD` 确认（**采用**） | 每次桶存储刷新 +2 RT（SCARD 管道 + HDEL），并省掉每键每 band 的 `SMEMBERS` | 写仍批量 | 每桶一个键，≤缓冲容量 | 完全照旧 |
| C. 会话级集合 | 0 | 写仍批量 | 随会话总长无界增长 | 照旧 |

以 10 万键、b=2 计：A 的删除阶段约 40 万次读往返；B 在 buffer=64 时约 0.94 万批次 +
20 万次必要读（`HEXISTS`+`LRANGE`，与既有代码相同），在 buffer=4096 时批次降到 150
左右。B 在任何缓冲大小下往返都显著更少，簿记更小，且不触碰查询可见性。

## 4. 验证

### 修复前后对照（fakeredis 执行 Redis 命令）

- 修复前：`scratch/baseline_output.txt`（重复插入被接受、删后幽灵命中、零成员桶条目、
  重复删除被接受、`is_empty()` 为 False）。
- 修复后：`scratch/final_check_repro.txt`——Redis 与 dict 行为一致：第二次插入
  `ValueError`；`remove` 后两种查询均空；桶统计无零成员条目；重复删除报"键不存在"；
  全删后 `is_empty()` 为 True。

### 测试套件（Python 3.12.16）

- 新增 `test/test_lsh_maintenance_sessions.py`：dict / fakeredis（prepickle 与
  bytes 键）/ 真实 redis-server（fixture 自起专用实例、随机 basename 命名空间）。
  独立维护键→签名预期表，校验同名键同/异签名、已落地与未刷出键的重复拒绝、重复删除、
  共享桶与最后成员、出错后继续操作、空会话、重复 close、关闭后调用、显式 close 与
  with 等价、同 basename 新对象重读、非法键/向量的既有异常、`check_duplication=False`
  原语义、簿记有界、后端失败传播；buffer_size ∈ {1,2,3,64,4096} 接纳结果一致。
- 新增 `test/test_lsh_session_benchmark.py`：批处理代价观测（见下）。
- 既有测试：`test_lsh.py`、`test_integration.py`（`DO_TEST_REDIS=true` 对真实
  Redis）、minhash/forest/ensemble 等全部通过；全量 348 passed，仅剩与本次无关的
  既有环境问题（缺 pybloomfilter、flaky 插件不兼容、缺 motor——在未修改基线上相同）。

### 10 万键批处理观测（`DATASKETCH_BENCH_N=100000`，逐键生成，测试端不预存 sketch）

插入→删除→再插入，自动刷新边界前后安排重复请求（606 次重复请求全部被拒，两种缓冲
大小结果一致）。fakeredis 与真实 Redis 的批次数/命令数/往返**完全一致**：

| 后端 | buffer | 阶段 | 批次 | 命令 | 往返 | Python 峰值 | 簿记峰值 | 关闭后残留 |
|---|---|---|---|---|---|---|---|---|
| fakeredis | 64 | insert | 9,375 | 700,100 | 109,475 | 148.45 MB* | ≤64 | 无 |
| fakeredis | 64 | delete | 9,377 | 801,680 | 211,057 | 0.19 MB | ≤64 | 无 |
| fakeredis | 4096 | insert | 147 | 700,098 | 100,245 | 161.88 MB* | ≤4096 | 无 |
| fakeredis | 4096 | delete | 149 | 725,124 | 200,273 | 6.24 MB | ≤4096 | 无 |
| real redis | 64 | insert | 9,375 | 700,100 | 109,475 | 0.33 MB | ≤64 | 无 |
| real redis | 64 | delete | 9,377 | 801,680 | 211,057 | 0.30 MB | ≤64 | 无 |
| real redis | 4096 | insert | 147 | 700,098 | 100,245 | 4.23 MB | ≤4096 | 无 |
| real redis | 4096 | delete | 149 | 725,124 | 200,273 | 4.77 MB | ≤4096 | 无 |

（re-insert 与 insert 相同，略。*fakeredis 的峰值包含进程内模拟服务器持有的全部
"数据库"状态；真实 Redis 下客户端侧额外峰值仅 0.3–4.8 MB，且随缓冲容量而非键数增长。）

- 批次数 ≈ 3 个存储 × 2N/buffer（写全部成批，**没有退化成逐键写回**：buffer=64 时
  每批约 32 键/存储；buffer=4096 时 10 万键仅 147 批）。
- 删除阶段因刷新时确认桶目录，每次桶存储刷新多 1 个 SCARD 管道（已计入），但省掉了
  旧代码每键每 band 的 `SMEMBERS`（10 万键 b=2 即省 20 万往返）。
- 关闭后 `_pending_*` 全空、`command_stack` 全空，fakeredis 命名空间键数精确等于
  2N+1003（1 键目录 + N 签名列表 + 2 桶目录 + 1000+N 个桶集）。

## 5. 复跑命令

```bash
# 环境（Python 3.12 经 uv 安装；redis-server 7.2.5 源码编译于 /tmp/redis-build）
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e . \
  redis fakeredis pytest pytest-cov pytest-asyncio mockredispy flaky numpy scipy

# 修复前基线（在原始提交上）：python scratch/repro_baseline.py 等，输出见
#   scratch/baseline_output.txt
# 修复后复现：
.venv/bin/python scratch/repro_baseline.py
.venv/bin/python scratch/probe_deletion_baseline.py

# 新套件（fakeredis + dict + 真实 redis-server 集成）
REDIS_SERVER_BIN=/tmp/redis-build/redis-7.2.5/src/redis-server \
  .venv/bin/python -m pytest test/test_lsh_maintenance_sessions.py -q

# 批处理观测（默认 3000 键；10 万键见 scratch/bench_*_100k.log）
REDIS_SERVER_BIN=... .venv/bin/python -m pytest test/test_lsh_session_benchmark.py -q -s
DATASKETCH_BENCH_N=100000 REDIS_SERVER_BIN=... \
  .venv/bin/python -m pytest test/test_lsh_session_benchmark.py -q -s

# 既有测试
DO_TEST_REDIS=true .venv/bin/python -m pytest test/test_lsh.py test/test_integration.py -q
.venv/bin/python -m pytest test/ -q --ignore=test/aio
```

实际输出：`scratch/final_test_run.txt`（280 passed, 30 skipped——LSH/存储/会话相关），
全量 `348 passed`（8 failed + 4 errors 为基线既有环境问题）。

## 6. 变更清单

- `datasketch/storage.py`：`RedisBuffer` 执行钩子与失败统一丢弃；`Storage.buffer_size`
  传播到自有写缓冲（修 MRO 遮蔽）；`RedisListStorage` 待处理键簿记与 `has_key` 覆盖；
  `RedisSetStorage` 刷新时空桶确认；`empty_buffer` 空栈跳过重连；`remove_val` 的
  `hdel` 字段名与 `lrem` 参数修正；`deferred_bucket_cleanup` 能力标志。
- `datasketch/lsh.py`：`_MinHashLSHSession` 基类（幂等 close、失败传播、关闭后拒绝）、
  `_remove` 对延迟清理存储跳过逐键空桶探测、会话文档字符串。
- `docs/lsh.rst`：新增 "Maintenance sessions" 一节（能接受哪些调用、何时可见、出错
  表示什么）。
- `test/test_lsh_maintenance_sessions.py`、`test/test_lsh_session_benchmark.py`：新增。
- `pyproject.toml`：注册 `integration` 标记。

未动：Cassandra 与 aio 路径（同类缓冲问题存在但无法在本环境验证，aio-redis 会随基类
修复受益）；既有损坏数据的修复；分片合并接口；`check_duplication=False` 语义。
