from __future__ import annotations

import json
from pathlib import Path


TARGET_PATH = Path("data/tutoring_question_bank/questions.jsonl")
TARGET_PER_SUBJECT = 100


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_record(
    *,
    qid: str,
    subject_id: str,
    problem_type: str,
    knowledge_points: list[str],
    difficulty: str,
    question: str,
    answer: str,
    solution_steps: list[str],
    common_mistakes: list[str],
) -> dict:
    return {
        "id": qid,
        "subject_id": subject_id,
        "problem_type": problem_type,
        "knowledge_points": knowledge_points,
        "difficulty": difficulty,
        "question": question,
        "answer": answer,
        "solution_steps": solution_steps,
        "common_mistakes": common_mistakes,
    }


def c_question(num: int) -> dict:
    case = (num - 11) % 6
    qid = f"c_{num:03d}"
    if case == 0:
        start = (num % 4) + 1
        end = start + (num % 5) + 2
        total = sum(range(start, end + 1))
        return make_record(
            qid=qid,
            subject_id="C_program",
            problem_type="c_output",
            knowledge_points=["循环", "变量跟踪", "printf"],
            difficulty="easy",
            question=(
                f"阅读代码：int i, s = 0; for (i = {start}; i <= {end}; i++) "
                f"s += i; printf(\"%d\", s); 程序输出什么？"
            ),
            answer=f"输出 {total}。",
            solution_steps=[
                f"循环变量 i 从 {start} 开始，到 {end} 结束。",
                "每轮把当前 i 累加到 s 中。",
                f"累加和为 {start}+...+{end}={total}。",
                f"最终 printf 输出 {total}。",
            ],
            common_mistakes=["把循环终止后的 i 也加入求和。", "忽略 s 的初值为 0。"],
        )
    if case == 1:
        values = [num % 7 + 1, num % 7 + 3, num % 7 + 5, num % 7 + 7]
        index = num % 4
        return make_record(
            qid=qid,
            subject_id="C_program",
            problem_type="c_output",
            knowledge_points=["数组", "下标", "printf"],
            difficulty="easy",
            question=(
                f"阅读代码：int a[4] = {{{values[0]}, {values[1]}, {values[2]}, {values[3]}}}; "
                f"printf(\"%d\", a[{index}]); 程序输出什么？"
            ),
            answer=f"输出 {values[index]}。",
            solution_steps=[
                "C 数组下标从 0 开始。",
                f"a[{index}] 对应数组中的第 {index + 1} 个元素。",
                f"该元素的值为 {values[index]}，所以输出 {values[index]}。",
            ],
            common_mistakes=["把下标当成从 1 开始。", "把数组长度 4 和最大下标 3 混淆。"],
        )
    if case == 2:
        base = (num % 6) * 5 + 10
        values = [base, base + 10, base + 20, base + 30]
        offset = num % 2
        delta = 1 + (num % 2)
        target = offset + delta
        return make_record(
            qid=qid,
            subject_id="C_program",
            problem_type="c_pointer_array",
            knowledge_points=["指针", "数组", "指针算术"],
            difficulty="medium",
            question=(
                f"阅读代码：int a[4] = {{{values[0]}, {values[1]}, {values[2]}, {values[3]}}}; "
                f"int *p = &a[{offset}]; p += {delta}; printf(\"%d\", *p); 输出什么？"
            ),
            answer=f"输出 {values[target]}。",
            solution_steps=[
                f"p 初始指向 a[{offset}]。",
                f"p += {delta} 后，p 指向 a[{target}]。",
                f"*p 读取 a[{target}] 的值，即 {values[target]}。",
            ],
            common_mistakes=["把 p+=1 理解成地址加 1 字节。", "混淆 p 和 *p。"],
        )
    if case == 3:
        x = (num % 9) + 1
        delta = (num % 4) + 1
        result = x + delta
        return make_record(
            qid=qid,
            subject_id="C_program",
            problem_type="c_function_call",
            knowledge_points=["指针参数", "函数调用", "间接修改"],
            difficulty="medium",
            question=(
                f"阅读代码：void add(int *p) {{ *p += {delta}; }} "
                f"int main(void) {{ int x = {x}; add(&x); printf(\"%d\", x); }} 输出什么？"
            ),
            answer=f"输出 {result}。",
            solution_steps=[
                "main 中把 x 的地址传给函数 add。",
                "add 通过 *p 间接访问并修改 x。",
                f"x 从 {x} 增加 {delta}，变为 {result}。",
            ],
            common_mistakes=["把 *p += k 和 p += k 混淆。", "认为值传递一定不能修改调用者变量。"],
        )
    if case == 4:
        var = chr(ord("a") + (num % 4))
        return make_record(
            qid=qid,
            subject_id="C_program",
            problem_type="c_debug",
            knowledge_points=["scanf", "取地址", "未定义行为"],
            difficulty="easy",
            question=f"找错改错：int {var}; scanf(\"%d\", {var}); printf(\"%d\", {var}); 这段代码有什么问题？",
            answer=f"scanf 的第二个参数应传入变量地址，应改为 scanf(\"%d\", &{var});。",
            solution_steps=[
                "scanf 需要把输入写入变量所在内存。",
                f"要写入 {var}，必须传入它的地址 &{var}。",
                f"原代码把 {var} 的值当作地址传入，可能导致未定义行为。",
            ],
            common_mistakes=["忘记 scanf 参数通常要加 &。", "只检查 printf 而忽略输入语句。"],
        )
    word = ["abcd", "hello", "code", "exam"][num % 4]
    pos = num % len(word)
    new_char = chr(ord("x") - (num % 3))
    result = word[:pos] + new_char + word[pos + 1 :]
    return make_record(
        qid=qid,
        subject_id="C_program",
        problem_type="c_output",
        knowledge_points=["字符串", "字符数组", "下标"],
        difficulty="easy",
        question=f"阅读代码：char s[] = \"{word}\"; s[{pos}] = '{new_char}'; printf(\"%s\", s); 程序输出什么？",
        answer=f"输出 {result}。",
        solution_steps=[
            "s 是可修改的字符数组。",
            f"s[{pos}] 原来是字符 '{word[pos]}'。",
            f"把它改为 '{new_char}' 后，字符串变为 {result}。",
        ],
        common_mistakes=["把字符数组和不可修改的字符串字面量指针混淆。", "忘记字符串以下标访问字符。"],
    )


def page_faults(seq: list[int], frames: int, algorithm: str) -> int:
    memory: list[int] = []
    fifo_queue: list[int] = []
    last_used: dict[int, int] = {}
    faults = 0
    for t, page in enumerate(seq):
        if page in memory:
            last_used[page] = t
            continue
        faults += 1
        if len(memory) < frames:
            memory.append(page)
            fifo_queue.append(page)
            last_used[page] = t
            continue
        if algorithm == "FIFO":
            victim = fifo_queue.pop(0)
        elif algorithm == "LRU":
            victim = min(memory, key=lambda p: last_used.get(p, -1))
        else:
            future = seq[t + 1 :]
            victim = max(
                memory,
                key=lambda p: future.index(p) if p in future else 10**9,
            )
        memory[memory.index(victim)] = page
        if algorithm == "FIFO":
            fifo_queue.append(page)
        last_used[page] = t
    return faults


def fcfs(processes: list[tuple[str, int, int]]) -> tuple[list[str], dict[str, tuple[int, int]]]:
    time_now = 0
    order = []
    metrics = {}
    for name, arrival, service in sorted(processes, key=lambda item: (item[1], item[0])):
        start = max(time_now, arrival)
        finish = start + service
        order.append(name)
        metrics[name] = (finish - arrival, start - arrival)
        time_now = finish
    return order, metrics


def sjf(processes: list[tuple[str, int, int]]) -> tuple[list[str], dict[str, tuple[int, int]]]:
    remaining = sorted(processes, key=lambda item: (item[1], item[0]))
    time_now = 0
    order = []
    metrics = {}
    while remaining:
        ready = [item for item in remaining if item[1] <= time_now]
        if not ready:
            time_now = min(item[1] for item in remaining)
            ready = [item for item in remaining if item[1] <= time_now]
        name, arrival, service = min(ready, key=lambda item: (item[2], item[1], item[0]))
        remaining.remove((name, arrival, service))
        start = time_now
        finish = start + service
        order.append(name)
        metrics[name] = (finish - arrival, start - arrival)
        time_now = finish
    return order, metrics


def rr(processes: list[tuple[str, int, int]], quantum: int) -> tuple[list[str], dict[str, int]]:
    arrivals = sorted(processes, key=lambda item: (item[1], item[0]))
    remaining = {name: service for name, _arrival, service in arrivals}
    finish: dict[str, int] = {}
    queue: list[str] = []
    time_now = 0
    idx = 0
    timeline = []
    while len(finish) < len(arrivals):
        while idx < len(arrivals) and arrivals[idx][1] <= time_now:
            queue.append(arrivals[idx][0])
            idx += 1
        if not queue:
            time_now = arrivals[idx][1]
            continue
        name = queue.pop(0)
        run = min(quantum, remaining[name])
        start = time_now
        time_now += run
        timeline.append(f"{name}({start}-{time_now})")
        remaining[name] -= run
        while idx < len(arrivals) and arrivals[idx][1] <= time_now:
            queue.append(arrivals[idx][0])
            idx += 1
        if remaining[name] > 0:
            queue.append(name)
        else:
            finish[name] = time_now
    return timeline, finish


def os_question(num: int) -> dict:
    qid = f"os_{num:03d}"
    case = (num - 11) % 6
    if case == 0:
        sequences = [
            [1, 2, 3, 1, 4, 5, 1, 2],
            [2, 3, 2, 1, 5, 2, 4, 5],
            [0, 1, 2, 0, 3, 0, 4, 2],
            [4, 1, 2, 4, 5, 1, 2, 3],
        ]
        seq = sequences[num % len(sequences)]
        frames = 3 + (num % 2)
        algorithm = ["FIFO", "LRU", "OPT"][num % 3]
        faults = page_faults(seq, frames, algorithm)
        return make_record(
            qid=qid,
            subject_id="operating_systems",
            problem_type="os_page_replacement",
            knowledge_points=[algorithm, "页面置换", "缺页次数"],
            difficulty="medium",
            question=f"页面访问序列为 {','.join(map(str, seq))}，页框数为 {frames}，采用 {algorithm} 页面置换算法，求缺页次数。",
            answer=f"缺页次数为 {faults}。",
            solution_steps=[
                f"按 {algorithm} 算法逐个处理访问序列。",
                "页面已在页框中则命中，不增加缺页次数。",
                "页面不在页框中则缺页；页框满时按算法规则选择被置换页面。",
                f"逐项统计后，缺页次数为 {faults}。",
            ],
            common_mistakes=["把 FIFO、LRU、OPT 的置换依据混用。", "命中页面时错误地增加缺页次数。"],
        )
    if case == 1:
        processes = [
            ("P1", 0, 2 + (num % 4)),
            ("P2", 1 + (num % 2), 3 + (num % 3)),
            ("P3", 3, 1 + (num % 5)),
        ]
        order, metrics = fcfs(processes)
        return make_record(
            qid=qid,
            subject_id="operating_systems",
            problem_type="os_cpu_scheduling",
            knowledge_points=["FCFS", "周转时间", "等待时间"],
            difficulty="medium",
            question=f"进程 {processes} 采用 FCFS 调度，求调度顺序、周转时间和等待时间。",
            answer=f"调度顺序为 {' -> '.join(order)}；周转/等待时间为 {metrics}。",
            solution_steps=[
                "FCFS 按到达先后顺序选择进程。",
                "若 CPU 空闲且下一个进程尚未到达，则时间推进到该进程到达时刻。",
                "周转时间 = 完成时间 - 到达时间。",
                "等待时间 = 开始运行时间 - 到达时间。",
            ],
            common_mistakes=["把服务时间误当作周转时间。", "忽略进程到达时间造成的等待。"],
        )
    if case == 2:
        processes = [
            ("P1", 0, 5 + (num % 4)),
            ("P2", 1, 2 + (num % 4)),
            ("P3", 2, 1 + (num % 3)),
        ]
        order, metrics = sjf(processes)
        return make_record(
            qid=qid,
            subject_id="operating_systems",
            problem_type="os_cpu_scheduling",
            knowledge_points=["SJF", "非抢占", "等待时间"],
            difficulty="medium",
            question=f"进程 {processes} 采用非抢占式 SJF 调度，求调度顺序和等待时间。",
            answer=f"调度顺序为 {' -> '.join(order)}；周转/等待时间为 {metrics}。",
            solution_steps=[
                "非抢占式 SJF 在 CPU 空闲时，从已到达进程中选服务时间最短者。",
                "正在运行的进程不会因新进程到达而被抢占。",
                "等待时间 = 开始运行时间 - 到达时间。",
            ],
            common_mistakes=["把非抢占式 SJF 当成抢占式 SRTF。", "在进程未到达时就参与比较。"],
        )
    if case == 3:
        processes = [("P1", 0, 4 + (num % 4)), ("P2", 1, 2 + (num % 3)), ("P3", 2, 3)]
        quantum = 2 + (num % 2)
        timeline, finish = rr(processes, quantum)
        return make_record(
            qid=qid,
            subject_id="operating_systems",
            problem_type="os_cpu_scheduling",
            knowledge_points=["RR", "时间片", "完成时间"],
            difficulty="medium",
            question=f"进程 {processes} 采用时间片轮转 RR，时间片 q={quantum}，求调度过程和完成时间。",
            answer=f"调度过程为 {', '.join(timeline)}；完成时间为 {finish}。",
            solution_steps=[
                "按到达时间进入就绪队列。",
                f"每次最多运行一个时间片 q={quantum}。",
                "未完成进程放回就绪队列尾部，完成进程记录完成时间。",
            ],
            common_mistakes=["时间片结束后忘记把未完成进程放回队尾。", "忽略运行期间新到达的进程。"],
        )
    if case == 4:
        max_vec = (7 + num % 3, 5 + num % 2, 3 + num % 2)
        alloc = (num % 3, (num + 1) % 2, (num + 2) % 2)
        need = tuple(m - a for m, a in zip(max_vec, alloc))
        return make_record(
            qid=qid,
            subject_id="operating_systems",
            problem_type="os_banker",
            knowledge_points=["银行家算法", "Need矩阵", "资源分配"],
            difficulty="medium",
            question=f"银行家算法中某进程 Max={max_vec}，Allocation={alloc}，求该进程 Need 向量。",
            answer=f"Need = Max - Allocation = {need}。",
            solution_steps=[
                "银行家算法中 Need 表示进程还可能需要的最大资源量。",
                "逐类资源计算 Need = Max - Allocation。",
                f"代入数据得到 Need={need}。",
            ],
            common_mistakes=["把 Need 写成 Allocation - Max。", "只算总量，不逐资源类型计算。"],
        )
    topic = ["生产者消费者", "读者写者", "前驱关系", "互斥访问"][num % 4]
    return make_record(
        qid=qid,
        subject_id="operating_systems",
        problem_type="os_pv_sync",
        knowledge_points=["PV操作", "信号量", topic],
        difficulty="medium",
        question=f"PV 同步题：针对“{topic}”场景，说明应如何区分互斥信号量和同步信号量，并给出设计原则。",
        answer="互斥信号量保护共享临界资源，同步信号量表达事件先后关系；应先分析资源冲突和前驱约束，再确定信号量初值与 P/V 位置。",
        solution_steps=[
            "先找共享资源，若多个进程不能同时访问，则设置互斥信号量。",
            "再找先后关系，若某事件必须等待另一事件发生，则设置同步信号量。",
            "互斥信号量初值通常为 1，同步信号量初值按初始可用事件数设置。",
        ],
        common_mistakes=["把互斥和同步都用同一个信号量表达。", "把 P/V 操作放反导致死锁。"],
    )


def sec_question(num: int) -> dict:
    qid = f"sec_{num:03d}"
    case = (num - 11) % 8
    if case == 0:
        p = [23, 29, 31, 37][num % 4]
        g = [2, 3, 5][num % 3]
        a = 3 + (num % 9)
        b = 5 + (num % 11)
        public_a = pow(g, a, p)
        public_b = pow(g, b, p)
        key = pow(public_b, a, p)
        return make_record(
            qid=qid,
            subject_id="cybersec_lab",
            problem_type="cybersec_phenomenon_analysis",
            knowledge_points=["Diffie-Hellman", "密钥交换", "模运算"],
            difficulty="medium",
            question=f"DH 密钥交换实验：公开 p={p}, g={g}，甲私钥 a={a}，乙私钥 b={b}。求双方公开值和共享密钥。",
            answer=f"甲公开值 A={public_a}，乙公开值 B={public_b}，共享密钥 K={key}。",
            solution_steps=[
                f"甲计算 A=g^a mod p={g}^{a} mod {p}={public_a}。",
                f"乙计算 B=g^b mod p={g}^{b} mod {p}={public_b}。",
                f"共享密钥可由 B^a mod p 或 A^b mod p 得到，结果为 {key}。",
            ],
            common_mistakes=["忘记每一步取模。", "把私钥当作公开值发送。"],
        )
    if case == 1:
        alg = ["DES", "AES", "3DES"][num % 3]
        return make_record(
            qid=qid,
            subject_id="cybersec_lab",
            problem_type="general_problem",
            knowledge_points=[alg, "对称加密", "密钥"],
            difficulty="easy",
            question=f"简答题：为什么 {alg} 属于对称加密？它适合解决什么问题？",
            answer=f"{alg} 加密和解密使用同一共享密钥，因此属于对称加密；它适合高效加密大量数据。",
            solution_steps=[
                "先说明对称加密的定义：加密和解密使用同一密钥。",
                f"指出 {alg} 使用共享密钥进行加密和解密。",
                "补充优点是速度快，缺点是密钥分发需要安全通道。",
            ],
            common_mistakes=["把对称加密和非对称加密混淆。", "只写算法名称，不解释密钥使用方式。"],
        )
    if case == 2:
        return make_record(
            qid=qid,
            subject_id="cybersec_lab",
            problem_type="general_problem",
            knowledge_points=["混合加密", "会话密钥", "性能"],
            difficulty="medium",
            question="为什么安全通信常用非对称加密协商会话密钥，再用对称加密传输数据？",
            answer="这样可以同时利用非对称加密便于密钥分发、对称加密速度快的优点。",
            solution_steps=[
                "非对称加密适合在不安全信道中保护或协商会话密钥。",
                "对称加密计算开销小，适合大量数据传输。",
                "混合加密把两者结合，兼顾安全性和效率。",
            ],
            common_mistakes=["认为混合加密只是重复加密。", "忽略会话密钥的作用。"],
        )
    if case == 3:
        return make_record(
            qid=qid,
            subject_id="cybersec_lab",
            problem_type="general_problem",
            knowledge_points=["数字签名", "完整性", "身份认证"],
            difficulty="medium",
            question="数字签名为什么能验证消息完整性和发送者身份？",
            answer="发送者用私钥对消息摘要签名，接收者用对应公钥验证；若消息被改动或签名者不匹配，验证会失败。",
            solution_steps=[
                "先对消息计算摘要。",
                "发送者用私钥对摘要进行签名。",
                "接收者用公钥验证签名。",
                "验证通过说明消息未被篡改且签名来自对应私钥持有者。",
            ],
            common_mistakes=["认为数字签名会自动隐藏明文。", "把签名和加密的目标混淆。"],
        )
    if case == 4:
        role = ["学生", "教师", "管理员"][num % 3]
        return make_record(
            qid=qid,
            subject_id="cybersec_lab",
            problem_type="general_problem",
            knowledge_points=["访问控制", "RBAC", "最小权限"],
            difficulty="medium",
            question=f"在线考试系统中，为什么要限制“{role}”角色只能访问与职责相关的功能？",
            answer="这是最小权限原则的要求，可以降低越权访问、误操作和数据泄露风险。",
            solution_steps=[
                "先明确角色对应的业务职责。",
                "只授予完成职责所需的最小权限。",
                "在后端进行权限校验，而不仅是前端隐藏入口。",
            ],
            common_mistakes=["给所有登录用户相同权限。", "只做前端控制，不做后端授权校验。"],
        )
    if case == 5:
        signal = ["多次登录失败", "异常 IP 登录", "权限变更", "考试提交异常"][num % 4]
        return make_record(
            qid=qid,
            subject_id="cybersec_lab",
            problem_type="cybersec_phenomenon_analysis",
            knowledge_points=["安全审计", "日志分析", signal],
            difficulty="easy",
            question=f"实验现象分析：审计日志中出现“{signal}”，应重点检查哪些字段？",
            answer="应检查账号、时间、来源 IP、操作类型、失败原因、User-Agent、结果状态以及前后关联事件。",
            solution_steps=[
                "先确定时间范围和受影响账号。",
                "统计事件频率和来源分布。",
                "结合结果状态和前后事件判断是否异常。",
            ],
            common_mistakes=["只看单条日志，不看上下文。", "忽略来源 IP 和时间序列。"],
        )
    if case == 6:
        reason = ["密钥不一致", "IV 不一致", "填充方式不一致", "密文截断"][num % 4]
        return make_record(
            qid=qid,
            subject_id="cybersec_lab",
            problem_type="cybersec_phenomenon_analysis",
            knowledge_points=["解密失败", "参数一致性", reason],
            difficulty="medium",
            question=f"实验现象分析：加密通信解密失败，怀疑原因是“{reason}”，应如何验证？",
            answer="应固定明文和密钥做最小测试，逐项比对密钥、模式、IV、填充、编码和密文完整性。",
            solution_steps=[
                "先固定一组最小测试输入，排除业务数据干扰。",
                f"重点检查 {reason} 是否在加密端和解密端一致。",
                "再检查编码转换和密文传输是否破坏数据。",
            ],
            common_mistakes=["只怀疑算法实现，不检查参数。", "把二进制密文当普通字符串处理。"],
        )
    lab = ["加密机制", "身份认证", "访问控制", "安全审计"][num % 4]
    return make_record(
        qid=qid,
        subject_id="cybersec_lab",
        problem_type="cybersec_lab_steps",
        knowledge_points=["实验步骤", lab, "结果验证"],
        difficulty="medium",
        question=f"实验步骤题：围绕“{lab}”实验，实验报告应如何组织操作步骤和结果验证？",
        answer="可按实验目标、环境准备、关键配置、执行过程、结果观察、结果分析和安全注意事项组织。",
        solution_steps=[
            "写清实验目标和授权环境。",
            "列出关键配置和输入参数。",
            "按执行顺序记录操作步骤。",
            "给出预期结果与实际结果对比。",
            "分析失败原因和安全边界。",
        ],
        common_mistakes=["只写操作，不写验证标准。", "忽略授权环境和风险说明。"],
    )


def extend_bank(existing: list[dict]) -> list[dict]:
    by_id = {item["id"]: item for item in existing}
    generators = {
        "c": c_question,
        "os": os_question,
        "sec": sec_question,
    }
    for prefix, generator in generators.items():
        for num in range(1, TARGET_PER_SUBJECT + 1):
            qid = f"{prefix}_{num:03d}"
            if qid not in by_id:
                by_id[qid] = generator(num)

    def sort_key(item: dict) -> tuple[int, int]:
        prefix, raw_num = str(item["id"]).split("_", 1)
        order = {"c": 0, "os": 1, "sec": 2}.get(prefix, 99)
        return order, int(raw_num)

    return sorted(by_id.values(), key=sort_key)


def main() -> None:
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    questions = extend_bank(load_existing(TARGET_PATH))
    lines = [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in questions]
    TARGET_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(questions)} questions to {TARGET_PATH}")


if __name__ == "__main__":
    main()
