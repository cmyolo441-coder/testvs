"""deepkernel — Shared Mathematical, Bytecode VM, Spectral Matrix, Sequence DP,
Probabilistic Sketching, Kalman/PID Control, GF(257) Threshold Crypto & Graph Topology Primitives.

Provides 7 reusable, zero-dependency algorithmic engines for EnforceX modules without
duplicating code across files:
  1. `SovereignDeepKernelBytecodeVM` & `SovereignDeepKernelDSLCompiler`
  2. `SovereignDeepKernelMatrixEngine` (Gauss-Jordan, Power Iteration, Markov, PageRank)
  3. `SovereignDeepKernelSequenceAligner` (Aho-Corasick, Rabin-Karp, Levenshtein, Smith-Waterman)
  4. `SovereignDeepKernelProbabilisticVault` (Bloom Filter, Count-Min Sketch, HyperLogLog, WAL)
  5. `SovereignDeepKernelAdaptiveControlEngine` (Kalman Filter, PID, Page-Hinkley, Welch-t, UCB1, Pareto)
  6. `SovereignDeepKernelCryptoThresholdVault` (GF(257) Shamir Secret Sharing & Lamport Signatures)
  7. `SovereignDeepKernelGraphTopologyEngine` (Tarjan SCC, Dijkstra, Edmonds-Karp Max-Flow, Huffman MDL)

Self-test: ``python3 -m fullagent.enforcex.deepkernel``
"""

from __future__ import annotations
import collections
import hashlib
import heapq
import math as _ext_math
import re as _ext_re
import struct
import threading
from dataclasses import dataclass as _ext_dataclass, field as _ext_field
from typing import Any as _Any, Dict as _Dict, List as _List, Optional as _Optional, Sequence as _Sequence, Set as _Set, Tuple as _Tuple

class SovereignDeepKernelOpcode:
    """Bytecode opcodes for the deontic domain policy virtual machine."""
    NOP = 0x00
    PUSH_CONST = 0x01
    LOAD_VAR = 0x02
    STORE_VAR = 0x03
    DUP = 0x04
    POP = 0x05
    ADD = 0x10
    SUB = 0x11
    MUL = 0x12
    DIV = 0x13
    MOD = 0x14
    MIN = 0x15
    MAX = 0x16
    CMP_EQ = 0x20
    CMP_NE = 0x21
    CMP_LT = 0x22
    CMP_LE = 0x23
    CMP_GT = 0x24
    CMP_GE = 0x25
    LOGIC_AND = 0x30
    LOGIC_OR = 0x31
    LOGIC_NOT = 0x32
    LOGIC_IMPLIES = 0x33
    LOGIC_XOR = 0x34
    STR_CONTAINS = 0x40
    STR_STARTSWITH = 0x41
    STR_ENDSWITH = 0x42
    REGEX_SEARCH = 0x43
    STR_LEN = 0x44
    WORD_COUNT = 0x45
    LINE_COUNT = 0x46
    SHANNON_ENTROPY = 0x47
    JUMP = 0x50
    JUMP_IF_FALSE = 0x51
    JUMP_IF_TRUE = 0x52
    EMIT_VIOLATION = 0x60
    ACCUM_SCORE = 0x61
    DOMAIN_OP = 0x70
    HALT = 0xFF

    @_ext_dataclass(frozen=True)
    class Instruction:
        opcode: int
        operand: _Any = None
        comment: str = ''

        def to_dict(self) -> _Dict[str, _Any]:
            return {'opcode': self.opcode, 'operand': self.operand, 'comment': self.comment}

@_ext_dataclass
class SovereignDeepKernelVMState:
    """Execution state of the deontic policy bytecode virtual machine."""
    ip: int = 0
    stack: _List[_Any] = _ext_field(default_factory=list)
    registers: _Dict[str, _Any] = _ext_field(default_factory=dict)
    violations: _List[_Dict[str, _Any]] = _ext_field(default_factory=list)
    trace_steps: int = 0
    score_delta: float = 0.0
    halted: bool = False

    def to_dict(self) -> _Dict[str, _Any]:
        return {
            'ip': self.ip,
            'stack_depth': len(self.stack),
            'top_of_stack': self.stack[-1] if self.stack else None,
            'registers': dict(self.registers),
            'violations': list(self.violations),
            'trace_steps': self.trace_steps,
            'score_delta': round(self.score_delta, 4),
            'halted': self.halted,
        }

class SovereignDeepKernelBytecodeVM:
    """Deterministic stack-based bytecode virtual machine for deontic rules."""

    def __init__(self, max_steps: int = 4096) -> None:
        self.max_steps = max(64, int(max_steps))

    @staticmethod
    def _compute_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts: _Dict[str, int] = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        n = float(len(text))
        return -sum((c / n) * _ext_math.log2(c / n) for c in counts.values() if c > 0)

    def execute(self, program: _Sequence[SovereignDeepKernelOpcode.Instruction], context: _Optional[_Dict[str, _Any]] = None) -> SovereignDeepKernelVMState:
        state = SovereignDeepKernelVMState()
        if context:
            state.registers.update(context)
        prog = list(program or [])
        n = len(prog)
        OP = SovereignDeepKernelOpcode
        while 0 <= state.ip < n and not state.halted and state.trace_steps < self.max_steps:
            ins = prog[state.ip]
            state.trace_steps += 1
            op = ins.opcode
            arg = ins.operand
            state.ip += 1
            if op == OP.NOP:
                continue
            elif op == OP.PUSH_CONST:
                state.stack.append(arg)
            elif op == OP.LOAD_VAR:
                state.stack.append(state.registers.get(str(arg)))
            elif op == OP.STORE_VAR:
                val = state.stack.pop() if state.stack else None
                state.registers[str(arg)] = val
            elif op == OP.DUP:
                state.stack.append(state.stack[-1] if state.stack else None)
            elif op == OP.POP:
                if state.stack:
                    state.stack.pop()
            elif op in (OP.ADD, OP.SUB, OP.MUL, OP.DIV, OP.MOD, OP.MIN, OP.MAX):
                b = float(state.stack.pop() if state.stack else 0.0)
                a_val = float(state.stack.pop() if state.stack else 0.0)
                if op == OP.ADD:
                    state.stack.append(a_val + b)
                elif op == OP.SUB:
                    state.stack.append(a_val - b)
                elif op == OP.MUL:
                    state.stack.append(a_val * b)
                elif op == OP.DIV:
                    state.stack.append(a_val / b if b != 0.0 else 0.0)
                elif op == OP.MOD:
                    state.stack.append(a_val % b if b != 0.0 else 0.0)
                elif op == OP.MIN:
                    state.stack.append(min(a_val, b))
                elif op == OP.MAX:
                    state.stack.append(max(a_val, b))
            elif op in (OP.CMP_EQ, OP.CMP_NE, OP.CMP_LT, OP.CMP_LE, OP.CMP_GT, OP.CMP_GE):
                b = state.stack.pop() if state.stack else 0
                a_val = state.stack.pop() if state.stack else 0
                try:
                    if op == OP.CMP_EQ:
                        res = (a_val == b)
                    elif op == OP.CMP_NE:
                        res = (a_val != b)
                    elif op == OP.CMP_LT:
                        res = (a_val < b)
                    elif op == OP.CMP_LE:
                        res = (a_val <= b)
                    elif op == OP.CMP_GT:
                        res = (a_val > b)
                    else:
                        res = (a_val >= b)
                except TypeError:
                    res = False
                state.stack.append(bool(res))
            elif op == OP.LOGIC_AND:
                b = bool(state.stack.pop() if state.stack else False)
                a_val = bool(state.stack.pop() if state.stack else False)
                state.stack.append(a_val and b)
            elif op == OP.LOGIC_OR:
                b = bool(state.stack.pop() if state.stack else False)
                a_val = bool(state.stack.pop() if state.stack else False)
                state.stack.append(a_val or b)
            elif op == OP.LOGIC_NOT:
                a_val = bool(state.stack.pop() if state.stack else False)
                state.stack.append(not a_val)
            elif op == OP.LOGIC_IMPLIES:
                b = bool(state.stack.pop() if state.stack else False)
                a_val = bool(state.stack.pop() if state.stack else False)
                state.stack.append((not a_val) or b)
            elif op == OP.LOGIC_XOR:
                b = bool(state.stack.pop() if state.stack else False)
                a_val = bool(state.stack.pop() if state.stack else False)
                state.stack.append(a_val != b)
            elif op == OP.STR_CONTAINS:
                needle = str(state.stack.pop() if state.stack else '')
                hay = str(state.stack.pop() if state.stack else '')
                state.stack.append(needle.lower() in hay.lower())
            elif op == OP.STR_STARTSWITH:
                prefix = str(state.stack.pop() if state.stack else '')
                hay = str(state.stack.pop() if state.stack else '')
                state.stack.append(hay.strip().lower().startswith(prefix.lower()))
            elif op == OP.STR_ENDSWITH:
                suffix = str(state.stack.pop() if state.stack else '')
                hay = str(state.stack.pop() if state.stack else '')
                state.stack.append(hay.strip().lower().endswith(suffix.lower()))
            elif op == OP.REGEX_SEARCH:
                pat = str(state.stack.pop() if state.stack else '')
                hay = str(state.stack.pop() if state.stack else '')
                try:
                    state.stack.append(bool(_ext_re.search(pat, hay, _ext_re.I | _ext_re.M)))
                except _ext_re.error:
                    state.stack.append(False)
            elif op == OP.STR_LEN:
                hay = str(state.stack.pop() if state.stack else '')
                state.stack.append(len(hay))
            elif op == OP.WORD_COUNT:
                hay = str(state.stack.pop() if state.stack else '')
                state.stack.append(len(hay.split()))
            elif op == OP.LINE_COUNT:
                hay = str(state.stack.pop() if state.stack else '')
                state.stack.append(len(hay.splitlines()))
            elif op == OP.SHANNON_ENTROPY:
                hay = str(state.stack.pop() if state.stack else '')
                state.stack.append(self._compute_entropy(hay))
            elif op == OP.JUMP:
                state.ip = max(0, min(n, int(arg or 0)))
            elif op == OP.JUMP_IF_FALSE:
                cond = bool(state.stack.pop() if state.stack else False)
                if not cond:
                    state.ip = max(0, min(n, int(arg or 0)))
            elif op == OP.JUMP_IF_TRUE:
                cond = bool(state.stack.pop() if state.stack else False)
                if cond:
                    state.ip = max(0, min(n, int(arg or 0)))
            elif op == OP.EMIT_VIOLATION:
                meta = arg if isinstance(arg, dict) else {'rule_id': str(arg or 'VM-RULE'), 'severity': 'high'}
                state.violations.append(dict(meta))
            elif op == OP.ACCUM_SCORE:
                state.score_delta += float(arg or 0.0)
            elif op == OP.DOMAIN_OP:
                top = state.stack.pop() if state.stack else ''
                state.stack.append(f'{arg}:{len(str(top))}')
            elif op == OP.HALT:
                state.halted = True
        return state

class SovereignDeepKernelDSLCompiler:
    """Compiles declarative deontic policy expressions into SovereignDeepKernelOpcode bytecode."""

    def compile_rule_program(self, min_words: int = 1, max_words: int = 5000, forbidden_substring: str = '', required_regex: str = '') -> _List[SovereignDeepKernelOpcode.Instruction]:
        OP = SovereignDeepKernelOpcode
        ins: _List[SovereignDeepKernelOpcode.Instruction] = [
            OP.Instruction(OP.LOAD_VAR, 'response_text', 'load response_text'),
            OP.Instruction(OP.WORD_COUNT, None, 'count words'),
            OP.Instruction(OP.STORE_VAR, 'word_count', 'save word_count'),
            OP.Instruction(OP.LOAD_VAR, 'word_count', 'load word_count'),
            OP.Instruction(OP.PUSH_CONST, int(min_words), 'min_words bound'),
            OP.Instruction(OP.CMP_GE, None, 'word_count >= min_words'),
            OP.Instruction(OP.LOAD_VAR, 'word_count', 'load word_count'),
            OP.Instruction(OP.PUSH_CONST, int(max_words), 'max_words bound'),
            OP.Instruction(OP.CMP_LE, None, 'word_count <= max_words'),
            OP.Instruction(OP.LOGIC_AND, None, 'bounds_satisfied'),
            OP.Instruction(OP.STORE_VAR, 'bounds_ok', 'store bounds_ok'),
        ]
        if forbidden_substring:
            ins.extend([
                OP.Instruction(OP.LOAD_VAR, 'response_text', 'load text for forbidden check'),
                OP.Instruction(OP.PUSH_CONST, forbidden_substring, 'push forbidden substring'),
                OP.Instruction(OP.STR_CONTAINS, None, 'check substring present'),
                OP.Instruction(OP.LOGIC_NOT, None, 'must not contain substring'),
                OP.Instruction(OP.LOAD_VAR, 'bounds_ok', 'load previous ok'),
                OP.Instruction(OP.LOGIC_AND, None, 'combine with bounds_ok'),
                OP.Instruction(OP.STORE_VAR, 'bounds_ok', 'update bounds_ok'),
            ])
        if required_regex:
            ins.extend([
                OP.Instruction(OP.LOAD_VAR, 'response_text', 'load text for regex check'),
                OP.Instruction(OP.PUSH_CONST, required_regex, 'push required regex'),
                OP.Instruction(OP.REGEX_SEARCH, None, 'regex search'),
                OP.Instruction(OP.LOAD_VAR, 'bounds_ok', 'load previous ok'),
                OP.Instruction(OP.LOGIC_AND, None, 'combine with bounds_ok'),
                OP.Instruction(OP.STORE_VAR, 'bounds_ok', 'update bounds_ok'),
            ])
        ins.extend([
            OP.Instruction(OP.LOAD_VAR, 'bounds_ok', 'final compliance boolean'),
            OP.Instruction(OP.HALT, None, 'halt execution'),
        ])
        return ins

class SovereignDeepKernelMatrixEngine:
    """Pure-Python dense/sparse matrix algebra, Gauss-Jordan elimination, power-iteration
    eigenvalue solver, and Markov stationary distribution solver for deontic."""

    @staticmethod
    def zeros(rows: int, cols: int) -> _List[_List[float]]:
        return [[0.0 for _ in range(max(1, cols))] for _ in range(max(1, rows))]

    @staticmethod
    def identity(n: int) -> _List[_List[float]]:
        m = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            m[i][i] = 1.0
        return m

    @staticmethod
    def transpose(mat: _Sequence[_Sequence[float]]) -> _List[_List[float]]:
        if not mat or not mat[0]:
            return [[]]
        rows, cols = len(mat), len(mat[0])
        return [[float(mat[r][c]) for r in range(rows)] for c in range(cols)]

    @staticmethod
    def matmul(a_mat: _Sequence[_Sequence[float]], b_mat: _Sequence[_Sequence[float]]) -> _List[_List[float]]:
        if not a_mat or not b_mat or not a_mat[0] or not b_mat[0]:
            return [[0.0]]
        r_a, c_a = len(a_mat), len(a_mat[0])
        r_b, c_b = len(b_mat), len(b_mat[0])
        k_lim = min(c_a, r_b)
        out = [[0.0 for _ in range(c_b)] for _ in range(r_a)]
        for i in range(r_a):
            row_i = a_mat[i]
            out_i = out[i]
            for k in range(k_lim):
                aik = float(row_i[k])
                if aik == 0.0:
                    continue
                b_row = b_mat[k]
                for j in range(c_b):
                    out_i[j] += aik * float(b_row[j])
        return out

    @staticmethod
    def matvec(mat: _Sequence[_Sequence[float]], vec: _Sequence[float]) -> _List[float]:
        if not mat:
            return []
        out: _List[float] = []
        for row in mat:
            k_lim = min(len(row), len(vec))
            out.append(sum(float(row[k]) * float(vec[k]) for k in range(k_lim)))
        return out

    @staticmethod
    def solve_linear_system(a_mat: _Sequence[_Sequence[float]], b_vec: _Sequence[float]) -> _List[float]:
        """Solve A * x = b using Gaussian elimination with partial pivoting."""
        n = len(a_mat)
        if n == 0:
            return []
        aug = [[float(a_mat[i][j]) for j in range(n)] + [float(b_vec[i] if i < len(b_vec) else 0.0)] for i in range(n)]
        for col in range(n):
            pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot_row][col]) < 1e-12:
                continue
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            piv = aug[col][col]
            for j in range(col, n + 1):
                aug[col][j] /= piv
            for r in range(n):
                if r != col and abs(aug[r][col]) > 1e-12:
                    factor = aug[r][col]
                    for j in range(col, n + 1):
                        aug[r][j] -= factor * aug[col][j]
        return [aug[i][n] for i in range(n)]

    @classmethod
    def dominant_eigenpair(cls, mat: _Sequence[_Sequence[float]], iterations: int = 60) -> _Tuple[float, _List[float]]:
        """Compute dominant eigenvalue and eigenvector via Power Iteration."""
        n = len(mat)
        if n == 0:
            return 0.0, []
        v = [1.0 / _ext_math.sqrt(n) for _ in range(n)]
        eig = 0.0
        for _ in range(max(5, iterations)):
            w = cls.matvec(mat, v)
            norm = _ext_math.sqrt(sum(x * x for x in w))
            if norm < 1e-12:
                return 0.0, v
            v_next = [x / norm for x in w]
            eig = sum(v_next[i] * w[i] for i in range(n))
            v = v_next
        return eig, v

    @classmethod
    def markov_stationary_distribution(cls, transition_matrix: _Sequence[_Sequence[float]], steps: int = 80) -> _List[float]:
        """Compute stationary distribution pi = pi * P of a row-stochastic Markov chain."""
        n = len(transition_matrix)
        if n == 0:
            return []
        dist = [1.0 / n for _ in range(n)]
        for _ in range(max(10, steps)):
            nxt = [0.0 for _ in range(n)]
            for i in range(n):
                row = transition_matrix[i]
                row_sum = sum(max(0.0, float(x)) for x in row)
                if row_sum <= 1e-12:
                    for j in range(n):
                        nxt[j] += dist[i] / n
                else:
                    for j in range(min(n, len(row))):
                        nxt[j] += dist[i] * (max(0.0, float(row[j])) / row_sum)
            tot = sum(nxt)
            dist = [x / tot for x in nxt] if tot > 0 else dist
        return dist

    @classmethod
    def pagerank_centrality(cls, adj: _Sequence[_Sequence[float]], damping: float = 0.85, iterations: int = 50) -> _List[float]:
        n = len(adj)
        if n == 0:
            return []
        pr = [1.0 / n for _ in range(n)]
        teleport = (1.0 - damping) / n
        for _ in range(max(10, iterations)):
            nxt = [teleport for _ in range(n)]
            for u in range(n):
                out_deg = sum(1 for v in range(len(adj[u])) if adj[u][v] > 0)
                if out_deg == 0:
                    share = damping * pr[u] / n
                    for v in range(n):
                        nxt[v] += share
                else:
                    share = damping * pr[u] / out_deg
                    for v in range(min(n, len(adj[u]))):
                        if adj[u][v] > 0:
                            nxt[v] += share
            pr = nxt
        return pr

class SovereignDeepKernelSequenceAligner:
    """Multi-pattern Aho-Corasick automaton, Rabin-Karp rolling hash, Levenshtein
    optimal edit transcript, and Smith-Waterman local sequence alignment for deontic."""

    def __init__(self, patterns: _Optional[_Sequence[str]] = None) -> None:
        self.goto: _List[_Dict[str, int]] = [{}]
        self.output: _List[_List[str]] = [[]]
        self.fail: _List[int] = [0]
        if patterns:
            self.build_automaton(patterns)

    def build_automaton(self, patterns: _Sequence[str]) -> None:
        self.goto = [{}]
        self.output = [[]]
        self.fail = [0]
        for pat in (patterns or ()):
            p = str(pat or '').lower()
            if not p:
                continue
            state = 0
            for ch in p:
                if ch not in self.goto[state]:
                    new_state = len(self.goto)
                    self.goto.append({})
                    self.output.append([])
                    self.fail.append(0)
                    self.goto[state][ch] = new_state
                state = self.goto[state][ch]
            if p not in self.output[state]:
                self.output[state].append(p)
        queue: collections.deque[int] = collections.deque()
        for ch, nxt in self.goto[0].items():
            self.fail[nxt] = 0
            queue.append(nxt)
        while queue:
            curr = queue.popleft()
            for ch, nxt in self.goto[curr].items():
                queue.append(nxt)
                f = self.fail[curr]
                while f > 0 and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[nxt] = self.goto[f].get(ch, 0)
                for out_pat in self.output[self.fail[nxt]]:
                    if out_pat not in self.output[nxt]:
                        self.output[nxt].append(out_pat)

    def search_all(self, text: str) -> _List[_Tuple[int, str]]:
        s = (text or '').lower()
        state = 0
        matches: _List[_Tuple[int, str]] = []
        for idx, ch in enumerate(s):
            while state > 0 and ch not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)
            for pat in self.output[state]:
                matches.append((idx - len(pat) + 1, pat))
        return matches

    @staticmethod
    def rabin_karp_search(text: str, pattern: str, base: int = 257, mod: int = 1000000007) -> _List[int]:
        n, m = len(text), len(pattern)
        if m == 0 or m > n:
            return []
        high_pow = pow(base, m - 1, mod)
        p_hash = 0
        t_hash = 0
        for i in range(m):
            p_hash = (p_hash * base + ord(pattern[i])) % mod
            t_hash = (t_hash * base + ord(text[i])) % mod
        hits: _List[int] = []
        for i in range(n - m + 1):
            if p_hash == t_hash and text[i:i + m] == pattern:
                hits.append(i)
            if i < n - m:
                t_hash = (t_hash - ord(text[i]) * high_pow) % mod
                t_hash = (t_hash * base + ord(text[i + m])) % mod
                t_hash = (t_hash + mod) % mod
        return hits

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        a_str, b_str = s1[:512], s2[:512]
        if len(a_str) < len(b_str):
            a_str, b_str = b_str, a_str
        if not b_str:
            return len(a_str)
        prev = list(range(len(b_str) + 1))
        for i, ca in enumerate(a_str, 1):
            curr = [i]
            for j, cb in enumerate(b_str, 1):
                ins = curr[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (0 if ca == cb else 1)
                curr.append(min(ins, dele, sub))
            prev = curr
        return prev[-1]

    @staticmethod
    def smith_waterman_align(seq_a: str, seq_b: str, match_score: int = 2, mismatch_cost: int = -1, gap_cost: int = -1) -> _Tuple[int, str]:
        """Compute optimal local sequence alignment score and aligned substring via Smith-Waterman DP."""
        a_s, b_s = seq_a[:256], seq_b[:256]
        m, n = len(a_s), len(b_s)
        if m == 0 or n == 0:
            return 0, ''
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        best_score = 0
        best_pos = (0, 0)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                diag = dp[i - 1][j - 1] + (match_score if a_s[i - 1] == b_s[j - 1] else mismatch_cost)
                up = dp[i - 1][j] + gap_cost
                left = dp[i][j - 1] + gap_cost
                val = max(0, diag, up, left)
                dp[i][j] = val
                if val > best_score:
                    best_score = val
                    best_pos = (i, j)
        i, j = best_pos
        aligned: _List[str] = []
        while i > 0 and j > 0 and dp[i][j] > 0:
            if dp[i][j] == dp[i - 1][j - 1] + (match_score if a_s[i - 1] == b_s[j - 1] else mismatch_cost):
                aligned.append(a_s[i - 1])
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + gap_cost:
                i -= 1
            else:
                j -= 1
        aligned.reverse()
        return best_score, ''.join(aligned)

class SovereignDeepKernelProbabilisticVault:
    """Bloom Filter, Count-Min Sketch, HyperLogLog cardinality estimator, and
    transactional Write-Ahead Log (WAL) key-value store for deontic."""

    def __init__(self, bloom_bits: int = 2048, cm_width: int = 128, cm_depth: int = 4) -> None:
        self.bloom_bits = max(128, int(bloom_bits))
        self.bloom_array = bytearray((self.bloom_bits + 7) // 8)
        self.cm_width = max(16, int(cm_width))
        self.cm_depth = max(2, int(cm_depth))
        self.cm_table = [[0 for _ in range(self.cm_width)] for _ in range(self.cm_depth)]
        self.hll_p = 6  # 64 registers
        self.hll_m = 1 << self.hll_p
        self.hll_registers = [0 for _ in range(self.hll_m)]
        self._store: _Dict[str, _Any] = {}
        self._wal: _List[_Dict[str, _Any]] = []
        self._lock = threading.RLock()

    @staticmethod
    def _hash64(item: str, seed: int = 0) -> int:
        h = hashlib.blake2b(f'{seed}:{item}'.encode('utf-8', errors='replace'), digest_size=8).digest()
        return struct.unpack('>Q', h)[0]

    def bloom_add(self, item: str) -> None:
        for seed in (101, 211, 307, 401):
            bit = self._hash64(item, seed) % self.bloom_bits
            self.bloom_array[bit >> 3] |= (1 << (bit & 7))

    def bloom_contains(self, item: str) -> bool:
        for seed in (101, 211, 307, 401):
            bit = self._hash64(item, seed) % self.bloom_bits
            if not (self.bloom_array[bit >> 3] & (1 << (bit & 7))):
                return False
        return True

    def cm_increment(self, item: str, count: int = 1) -> int:
        est = 1 << 60
        for d in range(self.cm_depth):
            col = self._hash64(item, 500 + d * 37) % self.cm_width
            self.cm_table[d][col] += int(count)
            if self.cm_table[d][col] < est:
                est = self.cm_table[d][col]
        return est

    def cm_query(self, item: str) -> int:
        return min(self.cm_table[d][self._hash64(item, 500 + d * 37) % self.cm_width] for d in range(self.cm_depth))

    def hll_add(self, item: str) -> None:
        x = self._hash64(item, 997)
        idx = x & (self.hll_m - 1)
        w = x >> self.hll_p
        rho = 1
        while w > 0 and (w & 1) == 0 and rho <= (64 - self.hll_p):
            rho += 1
            w >>= 1
        if rho > self.hll_registers[idx]:
            self.hll_registers[idx] = rho

    def hll_estimate(self) -> float:
        alpha = 0.709
        inv_sum = sum(2.0 ** (-r) for r in self.hll_registers)
        raw = alpha * (self.hll_m ** 2) / max(1e-12, inv_sum)
        zeros = sum(1 for r in self.hll_registers if r == 0)
        if raw <= 2.5 * self.hll_m and zeros > 0:
            return self.hll_m * _ext_math.log(float(self.hll_m) / zeros)
        return raw

    def tx_put(self, key: str, value: _Any) -> str:
        with self._lock:
            prev = self._store.get(key)
            self._store[key] = value
            digest = hashlib.sha256(f'{len(self._wal)}:{key}:{value}'.encode('utf-8', errors='replace')).hexdigest()[:16]
            self._wal.append({'seq': len(self._wal), 'op': 'PUT', 'key': key, 'prev': prev, 'value': value, 'digest': digest})
            self.bloom_add(key)
            self.cm_increment(key, 1)
            self.hll_add(key)
            return digest

    def tx_rollback(self, steps: int = 1) -> int:
        with self._lock:
            undone = 0
            for _ in range(max(0, int(steps))):
                if not self._wal:
                    break
                entry = self._wal.pop()
                k = entry['key']
                if entry['prev'] is None:
                    self._store.pop(k, None)
                else:
                    self._store[k] = entry['prev']
                undone += 1
            return undone

class SovereignDeepKernelAdaptiveControlEngine:
    """1D Kalman Compliance Filter, Discrete PID Enforcement Controller,
    Page-Hinkley Change-Point Detector, Welch t-Test, and UCB1 Bandit for deontic."""

    def __init__(self, target_setpoint: float = 95.0) -> None:
        self.setpoint = float(target_setpoint)
        # Kalman state
        self.kf_x = float(target_setpoint)
        self.kf_p = 1.0
        self.kf_q = 0.05
        self.kf_r = 0.25
        # PID state
        self.kp = 0.45
        self.ki = 0.12
        self.kd = 0.08
        self.integral_err = 0.0
        self.prev_err = 0.0
        # Page-Hinkley state
        self.ph_n = 0
        self.ph_mean = 0.0
        self.ph_cum = 0.0
        self.ph_min = 0.0
        # UCB1 Multi-Armed Bandit
        self.arm_counts: _Dict[str, int] = {}
        self.arm_rewards: _Dict[str, float] = {}
        self.total_pulls = 0

    def kalman_update(self, measurement: float) -> _Tuple[float, float]:
        # Predict
        x_pred = self.kf_x
        p_pred = self.kf_p + self.kf_q
        # Update
        k_gain = p_pred / (p_pred + self.kf_r)
        self.kf_x = x_pred + k_gain * (float(measurement) - x_pred)
        self.kf_p = (1.0 - k_gain) * p_pred
        return self.kf_x, self.kf_p

    def pid_step(self, current_score: float, dt: float = 1.0) -> float:
        dt_safe = max(1e-6, float(dt))
        err = self.setpoint - float(current_score)
        self.integral_err = max(-200.0, min(200.0, self.integral_err + err * dt_safe))
        deriv = (err - self.prev_err) / dt_safe
        self.prev_err = err
        control = self.kp * err + self.ki * self.integral_err + self.kd * deriv
        return max(-100.0, min(100.0, control))

    def page_hinkley_step(self, value: float, delta: float = 0.05, threshold: float = 15.0) -> _Tuple[bool, float]:
        self.ph_n += 1
        self.ph_mean += (float(value) - self.ph_mean) / self.ph_n
        self.ph_cum += (float(value) - self.ph_mean - delta)
        if self.ph_cum < self.ph_min:
            self.ph_min = self.ph_cum
        stat = self.ph_cum - self.ph_min
        return (stat > threshold), stat

    def ucb1_select_arm(self, arms: _Sequence[str]) -> str:
        arm_list = list(arms or ['default'])
        for a_name in arm_list:
            if self.arm_counts.get(a_name, 0) == 0:
                return a_name
        best_arm = arm_list[0]
        best_val = -1e18
        for a_name in arm_list:
            c = self.arm_counts[a_name]
            mean_r = self.arm_rewards.get(a_name, 0.0) / c
            bonus = _ext_math.sqrt((2.0 * _ext_math.log(max(1, self.total_pulls))) / c)
            if mean_r + bonus > best_val:
                best_val = mean_r + bonus
                best_arm = a_name
        return best_arm

    def ucb1_record_reward(self, arm: str, reward: float) -> None:
        self.total_pulls += 1
        self.arm_counts[arm] = self.arm_counts.get(arm, 0) + 1
        self.arm_rewards[arm] = self.arm_rewards.get(arm, 0.0) + float(reward)

    @staticmethod
    def welch_t_statistic(sample_a: _Sequence[float], sample_b: _Sequence[float]) -> float:
        na, nb = len(sample_a), len(sample_b)
        if na < 2 or nb < 2:
            return 0.0
        ma = sum(sample_a) / na
        mb = sum(sample_b) / nb
        va = sum((x - ma) ** 2 for x in sample_a) / (na - 1)
        vb = sum((x - mb) ** 2 for x in sample_b) / (nb - 1)
        denom = _ext_math.sqrt(max(1e-12, (va / na) + (vb / nb)))
        return (ma - mb) / denom

    @staticmethod
    def pareto_frontier(candidates: _Sequence[_Dict[str, float]], maximize_keys: _Sequence[str]) -> _List[_Dict[str, float]]:
        front: _List[_Dict[str, float]] = []
        keys = list(maximize_keys or [])
        for i, cand in enumerate(candidates):
            dominated = False
            for j, other in enumerate(candidates):
                if i == j:
                    continue
                if all(other.get(k, 0.0) >= cand.get(k, 0.0) for k in keys) and any(other.get(k, 0.0) > cand.get(k, 0.0) for k in keys):
                    dominated = True
                    break
            if not dominated:
                front.append(cand)
        return front

class SovereignDeepKernelCryptoThresholdVault:
    """Finite-field GF(257) Shamir (k, n) threshold secret sharing via Lagrange
    interpolation and Lamport hash-based one-time signatures for deontic."""
    PRIME = 257

    @classmethod
    def _mod_inv(cls, a_val: int) -> int:
        return pow(a_val % cls.PRIME, cls.PRIME - 2, cls.PRIME)

    @classmethod
    def split_secret_bytes(cls, secret: bytes, threshold_k: int = 3, shares_n: int = 5, seed: bytes = b'enforcex-gf257') -> _List[_Tuple[int, _List[int]]]:
        k = max(2, min(int(threshold_k), int(shares_n)))
        n = max(k, int(shares_n))
        shares: _List[_Tuple[int, _List[int]]] = [(x, []) for x in range(1, n + 1)]
        for idx, byte_val in enumerate(secret):
            coeffs = [int(byte_val)]
            for deg in range(1, k):
                digest = hashlib.sha256(seed + struct.pack('>II', idx, deg)).digest()
                coeffs.append(1 + (digest[0] % (cls.PRIME - 1)))
            for x_idx in range(n):
                x = x_idx + 1
                y = 0
                x_pow = 1
                for c in coeffs:
                    y = (y + c * x_pow) % cls.PRIME
                    x_pow = (x_pow * x) % cls.PRIME
                shares[x_idx][1].append(y)
        return shares

    @classmethod
    def recover_secret_bytes(cls, shares: _Sequence[_Tuple[int, _Sequence[int]]]) -> bytes:
        if not shares:
            return b''
        length = len(shares[0][1])
        out = bytearray(length)
        xs = [int(s[0]) for s in shares]
        k = len(xs)
        for byte_idx in range(length):
            secret_val = 0
            for i in range(k):
                num = 1
                den = 1
                for j in range(k):
                    if i != j:
                        num = (num * (-xs[j])) % cls.PRIME
                        den = (den * (xs[i] - xs[j])) % cls.PRIME
                lagrange = (num * cls._mod_inv(den)) % cls.PRIME
                secret_val = (secret_val + int(shares[i][1][byte_idx]) * lagrange) % cls.PRIME
            out[byte_idx] = secret_val & 0xFF
        return bytes(out)

    @staticmethod
    def lamport_keygen(seed: bytes, bits: int = 32) -> _Tuple[_List[_Tuple[bytes, bytes]], _List[_Tuple[bytes, bytes]]]:
        priv: _List[_Tuple[bytes, bytes]] = []
        pub: _List[_Tuple[bytes, bytes]] = []
        for i in range(bits):
            sk0 = hashlib.sha256(seed + f':0:{i}'.encode()).digest()[:16]
            sk1 = hashlib.sha256(seed + f':1:{i}'.encode()).digest()[:16]
            pk0 = hashlib.sha256(sk0).digest()[:16]
            pk1 = hashlib.sha256(sk1).digest()[:16]
            priv.append((sk0, sk1))
            pub.append((pk0, pk1))
        return priv, pub

    @staticmethod
    def lamport_sign(message: bytes, priv_key: _Sequence[_Tuple[bytes, bytes]]) -> _List[bytes]:
        digest = hashlib.sha256(message).digest()
        sig: _List[bytes] = []
        for i in range(len(priv_key)):
            byte_val = digest[i >> 3]
            bit = (byte_val >> (i & 7)) & 1
            sig.append(priv_key[i][bit])
        return sig

    @staticmethod
    def lamport_verify(message: bytes, signature: _Sequence[bytes], pub_key: _Sequence[_Tuple[bytes, bytes]]) -> bool:
        if len(signature) != len(pub_key):
            return False
        digest = hashlib.sha256(message).digest()
        for i in range(len(pub_key)):
            byte_val = digest[i >> 3]
            bit = (byte_val >> (i & 7)) & 1
            if hashlib.sha256(signature[i]).digest()[:16] != pub_key[i][bit]:
                return False
        return True

class SovereignDeepKernelGraphTopologyEngine:
    """Tarjan Strongly Connected Components (SCC), Kahn parallel wave scheduler,
    Dijkstra shortest-path remediation, Edmonds-Karp Max-Flow/Min-Cut, and
    Huffman minimum-description-length (MDL) codec for deontic."""

    @staticmethod
    def tarjan_scc(graph: _Dict[str, _Sequence[str]]) -> _List[_List[str]]:
        index = 0
        indices: _Dict[str, int] = {}
        lowlink: _Dict[str, int] = {}
        stack: _List[str] = []
        on_stack: _Set[str] = set()
        sccs: _List[_List[str]] = []

        def strongconnect(v: str) -> None:
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)
            for w in graph.get(v, ()) :
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])
            if lowlink[v] == indices[v]:
                comp: _List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)

        for node in list(graph.keys()):
            if node not in indices:
                strongconnect(node)
        return sccs

    @staticmethod
    def dijkstra_shortest_path(graph: _Dict[str, _Sequence[_Tuple[str, float]]], start: str, goal: str) -> _Tuple[float, _List[str]]:
        dist: _Dict[str, float] = {start: 0.0}
        prev: _Dict[str, str] = {}
        pq: _List[_Tuple[float, str]] = [(0.0, start)]
        visited: _Set[str] = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == goal:
                break
            for v, weight in graph.get(u, ()) :
                nd = d + max(0.0, float(weight))
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if goal not in dist:
            return float('inf'), []
        path: _List[str] = []
        curr: _Optional[str] = goal
        while curr is not None:
            path.append(curr)
            curr = prev.get(curr)
        path.reverse()
        return dist[goal], path

    @staticmethod
    def edmonds_karp_max_flow(capacity: _Sequence[_Sequence[float]], source: int, sink: int) -> float:
        n = len(capacity)
        if n == 0 or source == sink:
            return 0.0
        res = [[float(capacity[i][j]) for j in range(n)] for i in range(n)]
        max_flow = 0.0
        while True:
            parent = [-1 for _ in range(n)]
            parent[source] = source
            queue: collections.deque[_Tuple[int, float]] = collections.deque([(source, float('inf'))])
            augment = 0.0
            while queue:
                u, flow = queue.popleft()
                if u == sink:
                    augment = flow
                    break
                for v in range(n):
                    if parent[v] == -1 and res[u][v] > 1e-12:
                        parent[v] = u
                        queue.append((v, min(flow, res[u][v])))
            if augment <= 1e-12:
                break
            max_flow += augment
            curr = sink
            while curr != source:
                p_node = parent[curr]
                res[p_node][curr] -= augment
                res[curr][p_node] += augment
                curr = p_node
        return max_flow

    @staticmethod
    def huffman_code_lengths(text: str) -> _Dict[str, int]:
        if not text:
            return {}
        freqs: _Dict[str, int] = {}
        for ch in text:
            freqs[ch] = freqs.get(ch, 0) + 1
        if len(freqs) == 1:
            only_ch = next(iter(freqs))
            return {only_ch: 1}
        heap: _List[_Tuple[int, int, _Any]] = []
        uid = 0
        for ch, f in freqs.items():
            uid += 1
            heapq.heappush(heap, (f, uid, ch))
        while len(heap) > 1:
            f1, _, n1 = heapq.heappop(heap)
            f2, _, n2 = heapq.heappop(heap)
            uid += 1
            heapq.heappush(heap, (f1 + f2, uid, (n1, n2)))
        lengths: _Dict[str, int] = {}
        def walk(node: _Any, depth: int) -> None:
            if isinstance(node, str):
                lengths[node] = max(1, depth)
            else:
                walk(node[0], depth + 1)
                walk(node[1], depth + 1)
        walk(heap[0][2], 0)
        return lengths


def _self_test() -> int:
    vm = SovereignDeepKernelBytecodeVM()
    comp = SovereignDeepKernelDSLCompiler()
    prog = comp.compile_rule_program(min_words=2, max_words=50, forbidden_substring="BAD_TOKEN", required_regex=r"verified")
    st_ok = vm.execute(prog, {"response_text": "All systems verified and operational."})
    assert st_ok.registers.get("bounds_ok") is True
    st_bad = vm.execute(prog, {"response_text": "Contains BAD_TOKEN here."})
    assert st_bad.registers.get("bounds_ok") is False

    mat = SovereignDeepKernelMatrixEngine()
    sol = mat.solve_linear_system([[2.0, 1.0], [1.0, 3.0]], [5.0, 5.0])
    assert abs(sol[0] - 2.0) < 1e-6 and abs(sol[1] - 1.0) < 1e-6
    pr = mat.pagerank_centrality([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    assert abs(sum(pr) - 1.0) < 1e-5

    seq = SovereignDeepKernelSequenceAligner(["obligatory", "forbidden"])
    assert len(seq.search_all("obligatory and forbidden norms")) == 2
    assert seq.rabin_karp_search("abracadabra", "abra") == [0, 7]
    assert seq.levenshtein_distance("kitten", "sitting") == 3
    sw_score, sw_sub = seq.smith_waterman_align("ACGTACGT", "TACGT")
    assert sw_score >= 8 and "TACGT" in sw_sub

    vault = SovereignDeepKernelProbabilisticVault()
    vault.tx_put("k1", "v1")
    vault.tx_put("k2", "v2")
    assert vault.bloom_contains("k1") and vault.cm_query("k1") >= 1
    assert vault.tx_rollback(1) == 1

    ctrl = SovereignDeepKernelAdaptiveControlEngine(target_setpoint=95.0)
    kf_val, _ = ctrl.kalman_update(92.0)
    assert 90.0 <= kf_val <= 96.0
    arm = ctrl.ucb1_select_arm(["strict", "balanced"])
    ctrl.ucb1_record_reward(arm, 1.0)

    crypto = SovereignDeepKernelCryptoThresholdVault()
    secret = b"SOVEREIGN-KEY-42"
    shares = crypto.split_secret_bytes(secret, threshold_k=3, shares_n=5)
    assert crypto.recover_secret_bytes(shares[:3]) == secret
    priv, pub = crypto.lamport_keygen(b"lamport-seed", bits=16)
    sig = crypto.lamport_sign(b"audit-msg", priv)
    assert crypto.lamport_verify(b"audit-msg", sig, pub) is True

    topo = SovereignDeepKernelGraphTopologyEngine()
    sccs = topo.tarjan_scc({"A": ["B"], "B": ["A", "C"], "C": []})
    assert any(len(c) == 2 for c in sccs)
    cost, dpath = topo.dijkstra_shortest_path({"S": [("M", 2.0), ('T', 9.0)], "M": [("T", 3.0)], "T": []}, "S", "T")
    assert cost == 5.0 and dpath == ["S", "M", "T"]
    assert topo.edmonds_karp_max_flow([[0, 10, 5, 0], [0, 0, 4, 8], [0, 0, 0, 7], [0, 0, 0, 0]], 0, 3) == 15.0

    print("deepkernel._self_test: OK PASS (BytecodeVM, Gauss-Jordan, Aho-Corasick, Smith-Waterman, Bloom/HLL, Kalman/PID, GF(257) Shamir, Tarjan/MaxFlow)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
