from tinygrad.ops import Ops, GroupOp, UOp, UPat, PatternMatcher, TrackedPatternMatcher, graph_rewrite
from tinygrad import dtypes
from tinygrad.dtype import DType
from typing import Optional
from tinygrad.renderer import Renderer
import math
from dataclasses import dataclass
from os import getenv

def twos(bits, val):
  if val < 0:
    val = -val
    val = (~val) & ((1 << bits) - 1)
    val += 1
  return val

class BinPat:
    def __init__(self, total, pat):
        """
        # example:
        BinPat(32, [("imm12", 7, (11,5)),
                    ("rs2",   5)
                    ("rs1",   5),
                    ("func3", 3),
                    ("imm12", 5, (4,0)),
                    ("opc",   7),
                    ])
        """

        self.total = total
        self.pat = pat
        if sum([p[1] for p in pat]) != total:
            raise ValueError("pat doesn't amount to toal")
        self.fields_total_len = {}
        self.fields_masks = {}
        for p in pat:
            pat_name = p[0]
            pat_len = p[1]
            pat_slice = (pat_len-1, 0)
            if len(p) >= 3:
                pat_slice = p[2]
            if pat_slice[0] != pat_slice[1] + pat_len - 1:
                raise ValueError(f"invalid slice spec for pat {pat_name}")

            v = self.fields_total_len.get(pat_name, 0)
            v += pat_len
            self.fields_total_len[pat_name] = v

            mask = ((1 << pat_len) - 1) << pat_slice[1]
            v = self.fields_masks.get(pat_name, 0)
            v = v | mask
            self.fields_masks[pat_name] = v


    def has_field(self, f):
        return f in self.fields_masks


    def __call__(self, **kwargs):
        out = 0
        for pat in self.pat:
            pat_name = pat[0]
            pat_len = pat[1]
            pat_slice = (pat_len-1, 0)
            if len(pat) >= 3:
                pat_slice = pat[2]

            v = kwargs.get(pat_name, None)
            if v is None:
                raise ValueError(f"pattern field {pat_name} is unset")
            v = twos(self.fields_total_len[pat_name], v)
            if v & self.fields_masks[pat_name] != v:
                raise ValueError(f"there are some bits in the val that don't fit into the pattern {pat_name}")
            v = v >> pat_slice[1]
            v = v & ((1 << pat_len) - 1)

            out = (out << pat_len) | v
        return out


    def decode(self, inp):
        outs = {}
        rpos = 0
        for pat in self.pat:
            pat_name = pat[0]
            pat_len = pat[1]
            pat_slice = (pat_len-1, 0)
            if len(pat) >= 3:
                pat_slice = pat[2]

            pos = self.total - rpos - pat_len
            b = (inp >> pos) & ((1 << pat_len) - 1)
            b = b << pat_slice[1]

            v = outs.get(pat_name, 0)
            v = v | b
            outs[pat_name] = v

            rpos += pat_len
        return outs


class Encodings:
    R = BinPat(32, [("func7",7),
                    ("rs2",  5),
                    ("rs1",  5),
                    ("func3",3),
                    ("rd",   5),
                    ("opc",  7),
                ])

    I = BinPat(32, [("imm",  12),
                    ("rs1",   5),
                    ("func3", 3),
                    ("rd",    5),
                    ("opc",   7),
                    ])

    # identical to I, except that the top 5 bits of imm are seperate
    I_shift = BinPat(32, [("shift", 5),
                          ("imm",   7),
                          ("rs1",   5),
                          ("func3", 3),
                          ("rd",    5),
                          ("opc",   7),
                          ])

    S = BinPat(32, [("imm",   7, (11,5)),
                    ("rs2",   5),
                    ("rs1",   5),
                    ("func3", 3),
                    ("imm",   5, (4,0)),
                    ("opc",   7),
                    ])

    B = BinPat(32, [("imm",   1, (12,12)),
                    ("imm",   6, (10,5)),
                    ("rs2",   5),
                    ("rs1",   5),
                    ("func3", 3),
                    ("imm",   4, (4,1)),
                    ("imm",   1, (11,11)),
                    ("opc",   7),
                    ])

    U = BinPat(32, [("imm", 20, (31,12)),
                    ("rd",  5),
                    ("opc", 7),
                    ])

    J = BinPat(32, [("imm", 1, (20,20)),
                    ("imm",10, (10, 1)),
                    ("imm", 1, (11,11)),
                    ("imm", 8, (19,12)),
                    ("rd",  5),
                    ("opc", 7),
                    ])

    # cursed way of dispatching tensix ops from a baby risc
    # see https://www.corsix.org/content/tt-wh-part5
    TENSIX = BinPat(32, [("tensix",30, (29, 0)),
                         ("tensix",2,  (31,30))
                         ])


class RvOp:
    def __init__(self, encoding, **require):
        self.require = require
        self.encoding = encoding
        self.unset = list(set([p[0] for p in encoding.pat if not p[0] in require]))

    def encode(self, **args):
        u = self.require | args
        return self.encoding(**u)

# missing ops rv32i (therefore also rv64i):
# - fence
# - fence.tso
# - pause
# - ecall
# - ebreak

# missing ops rv64i:
# - lwu
# - ld
# - sd
# - addiw, slliw, srliw, sraiw
# - addw, subw, sllw, srlw, sraw
# - add.uw, sh1add.uw, sh2add.uw, sh3add.uw, slli.uw  (for Zba)

# resources:
#   official spec:           https://riscv.github.io/riscv-isa-manual/snapshot/unprivileged/#rv32-64g 
#   subset, easier to read:  https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html
#   encoder/decoder tool:    https://luplab.gitlab.io/rvcodecjs/

class RvOps:
    """
    All ops operate on XLEN (32 bits for rv32i, 64 bits for rv64i), unless otherwise stated.
    Some ops have a bit different behaviour (currently only all shift related ops) on rv64i
    """

    def op(encoding, **require):
        return RvOp(encoding, **require)

    # rd = sext(imm)
    LUI   = op(Encodings.U, opc=0b0110111)

    # rd = pc + sext(imm)
    AUIPC = op(Encodings.U, opc=0b0010111)

    # rd = rs1 + sext(imm)
    ADDI  = op(Encodings.I, func3=0b000, opc=0b0010011)

    # rd = rs1 s< sext(imm)
    SLTI  = op(Encodings.I, func3=0b010, opc=0b0010011)

    # rd = rs1 u< sext(imm)
    SLTIU = op(Encodings.I, func3=0b011, opc=0b0010011)

    # rd = rs1 ^ sext(imm)
    XORI  = op(Encodings.I, func3=0b100, opc=0b0010011)

    # rd = rs1 | sext(imm)
    ORI   = op(Encodings.I, func3=0b110, opc=0b0010011)

    # rd = rs1 & sext(imm)
    ANDI  = op(Encodings.I, func3=0b111, opc=0b0010011)

    # in rv32:
    #   rd = rs1 << (imm &  b11111)
    # in rv64:
    #   rd = rs1 << (imm & b111111)
    SLLI  = op(Encodings.I_shift, shift=0b00000, func3=0b001, opc=0b0010011)

    # in rv32:
    #   rd = rs1 >> (imm &  b11111)
    # in rv64:
    #   rd = rs1 >> (imm & b111111)
    SRLI  = op(Encodings.I_shift, shift=0b00000, func3=0b101, opc=0b0010011)

    # in rv32:
    #   rd = rs1 s>> (imm &  b11111)
    # in rv64:
    #   rd = rs1 s>> (imm & b111111)
    SRAI  = op(Encodings.I_shift, shift=0b01000, func3=0b101, opc=0b0010011)

    # rd = rs1 + rs2
    ADD   = op(Encodings.R, func7=0b0000000, func3=0b000, opc=0b0110011)

    # rd = rs1 - rs2
    SUB   = op(Encodings.R, func7=0b0100000, func3=0b000, opc=0b0110011)

    # in rv32:
    #   rd = rs1 << (rs2 &  b11111)
    # in rv64:
    #   rd = rs1 << (rs2 & b111111)
    SLL   = op(Encodings.R, func7=0b0000000, func3=0b001, opc=0b0110011)

    # rd = rs1 s< rs2
    SLT   = op(Encodings.R, func7=0b0000000, func3=0b010, opc=0b0110011)

    # rd = rs1 u< rs2
    SLTU  = op(Encodings.R, func7=0b0000000, func3=0b011, opc=0b0110011)

    # rd = rs1 ^ rs2
    XOR   = op(Encodings.R, func7=0b0000000, func3=0b100, opc=0b0110011)

    # in rv32:
    #   rd = rs1 >> (rs2 &  b11111)
    # in rv64:
    #   rd = rs1 >> (rs2 & b111111)
    SRL   = op(Encodings.R, func7=0b0000000, func3=0b101, opc=0b0110011)

    # in rv32:
    #   rd = rs1 s>> (rs2 &  b11111)
    # in rv64:
    #   rd = rs1 s>> (rs2 & b111111)
    SRA   = op(Encodings.R, func7=0b0100000, func3=0b101, opc=0b0110011)

    # rd = rs1 | rs2
    OR    = op(Encodings.R, func7=0b0000000, func3=0b110, opc=0b0110011)

    # rd = rs1 & rs2
    AND   = op(Encodings.R, func7=0b0000000, func3=0b111, opc=0b0110011)

    # rd = pc + 4
    # pc += sext(offset)
    JAL   = op(Encodings.J, opc=0b1101111)

    # t = pc + 4
    # pc = rs1 + sext(offset)
    # rd = t
    JALR  = op(Encodings.I, func3=0b000, opc=0b1100111)

    # if (rs1 == rs2)
    #   pc += sext(offset)
    BEQ   = op(Encodings.B, func3=0b000, opc=0b1100011)

    # if (rs1 != rs2)
    #   pc += sext(offset)
    BNE   = op(Encodings.B, func3=0b001, opc=0b1100011)

    # if (rs1 s< rs2)
    #   pc += sext(offset)
    BLT   = op(Encodings.B, func3=0b100, opc=0b1100011)

    # if (rs1 s>= rs2)
    #   pc += sext(offset)
    BGE   = op(Encodings.B, func3=0b101, opc=0b1100011)

    # if (rs1 u< rs2)
    #   pc += sext(offset)
    BLTU  = op(Encodings.B, func3=0b110, opc=0b1100011)

    # if (rs1 u>= rs2)
    #   pc += sext(offset)
    BGEU  = op(Encodings.B, func3=0b111, opc=0b1100011)

    # rd = sext((u8) mem[rs1 + sext(offset)])
    LB    = op(Encodings.I, func3=0b000, opc=0b0000011)

    # rd = sext((u16) mem[rs1 + sext(offset)])
    LH    = op(Encodings.I, func3=0b001, opc=0b0000011)

    # rd = sext((xlen) mem[rs1 + sext(offset)])
    LW    = op(Encodings.I, func3=0b010, opc=0b0000011)

    # rd = (u8) mem[rs1 + sext(offset)]
    LBU   = op(Encodings.I, func3=0b100, opc=0b0000011)

    # rd = (u16) mem[rs1 + sext(offset)]
    LHU   = op(Encodings.I, func3=0b101, opc=0b0000011)

    # mem[rs1 + sext(offset)] = (u8) rs2
    SB    = op(Encodings.S, func3=0b000, opc=0b0100011)

    # mem[rs1 + sext(offset)] = (u16) rs2
    SH    = op(Encodings.S, func3=0b001, opc=0b0100011)

    # mem[rs1 + sext(offset)] = (xlen) rs2
    SW    = op(Encodings.S, func3=0b010, opc=0b0100011)


    # ==== M extension ====

    # rd = (signed xlen) rs1 * (signed xlen) rs2
    MUL   = op(Encodings.R, func7=0b0000001, func3=0b000, opc=0b0110011)

    # upper part of MUL
    # rd = ((signed xlen) rs1 * (signed xlen) rs2) s>> xlen
    MULH  = op(Encodings.R, func7=0b0000001, func3=0b001, opc=0b0110011)

    # upper part of MUL
    # rd = ((signed xlen) rs1 * (un-signed xlen) rs2) s>> xlen
    MULHSU= op(Encodings.R, func7=0b0000001, func3=0b010, opc=0b0110011)

    # upper part of MUL
    # rd = ((un-signed xlen) rs1 * (un-signed xlen) rs2) u>> xlen
    MULHU = op(Encodings.R, func7=0b0000001, func3=0b011, opc=0b0110011)

    # rd = rs1 s/ rs2
    DIV   = op(Encodings.R, func7=0b0000001, func3=0b100, opc=0b0110011)

    # rd = rs1 u/ rs2
    DIVU  = op(Encodings.R, func7=0b0000001, func3=0b101, opc=0b0110011)

    # rd = rs1 s% rs2
    REM   = op(Encodings.R, func7=0b0000001, func3=0b110, opc=0b0110011)

    # rd = rs1 u% rs2
    REMU  = op(Encodings.R, func7=0b0000001, func3=0b111, opc=0b0110011)


    # ==== Zba extension ====
    
    # rd = (rs1 << 1) + rs2
    SH1ADD= op(Encodings.R, func7=0b0010000, func3=0b010, opc=0b0110011)

    # rd = (rs1 << 2) + rs2
    SH2ADD= op(Encodings.R, func7=0b0010000, func3=0b100, opc=0b0110011)

    # rd = (rs1 << 3) + rs2
    SH3ADD= op(Encodings.R, func7=0b0010000, func3=0b110, opc=0b0110011)


    # ==== TT Tensix baby risc ====
    TENSIX= op(Encodings.TENSIX)


class RvRegs:
    ra = 1 # return address
    sp = 2
    gp = 3 # global pointer
    tp = 4 # thread pointer

    caller_saved = [1, 5,6,7, 10,11,12,13,14,15,16,17, 28,29,30,31]
    callee_saved = [2, 8,9, 18,19,20,21,22,23,24,25,26,27]


    aliases = {x:i for i,x in enumerate([
        "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
        "s0","s1",
    ])}

    for k,v in {"fp":8, "t3":28, "t4":29,"t5":30, "t6":31}.items():
        aliases[k] = v

    for i in range(8):
        aliases[f"a{i}"] = i + 10

    for i in range(12):
        if i >= 2:
            aliases[f"s{i}"] = i + 18 - 2

    for i in range(32):
        aliases[f"x{i}"] = i


all_rv_ops = {attr.lower():getattr(RvOps,attr) for attr in dir(RvOps) if not callable(getattr(RvOps, attr)) and not attr.startswith("__")}
all_rv_ops_with_reg_dest = {x for x,v in all_rv_ops.items() if v.encoding.has_field("rd")}

class AsmReloc:
    def __init__(self, reloc_id, at):
        self.reloc_id = reloc_id
        self.at = at
        self.resolved = False

    def mark_resolved(self):
        if self.resolved:
            raise ValueError("reloc was already resolved")
        self.resolved = True

class AsmWriter:
    def __init__(self):
        self._next_reloc_id = 0
        self._bytes = []
        self._relocs = []

    def bytes(self, by):
        for b in by:
            self._bytes.append(b)

    # mark the following b{eq,ne,lt{,u},ge{,u}} as branch to unresolved label
    # returns a AsmReloc object, which has to be resolved manually later
    def reloc(self) -> AsmReloc:
        r = AsmReloc(self._next_reloc_id, self.addr)
        self._next_reloc_id += 1
        self._relocs.append(r)
        return r

    def op(self, op_nam, *args):
        args = list(args)
        st = f"{op_nam} {", ".join([str(x) for x in args])}"
        op = all_rv_ops[op_nam]
        vals = {}
        order = ["rd", "rs1", "rs2", "imm", "tensix"]
        for field in order:
            if op.encoding.has_field(field):
                a = args[0]
                if isinstance(a, str):
                    a = RvRegs.aliases[a]
                vals[field] = a
                args.pop(0)
        assert len(args) == 0
        self.bytes(op.encode(**vals).to_bytes(4, "little"))
        return st

    def addr(self):
        return len(self._bytes)

    # make the given reloc from earlier point to the given address
    def resolve(self, addr, reloc: AsmReloc):
        b = self._bytes[reloc.at : reloc.at+4]
        b = int.from_bytes(b, byteorder="little")
        b = Encodings.B.decode(b)
        b["imm"] = addr - reloc.at
        b = Encodings.B(**b).to_bytes(4, "little")
        self._bytes[reloc.at : reloc.at+4] = b
        reloc.mark_resolved()
        self._relocs.remove(reloc)

    def finish(self):
        if len(self._relocs) > 0:
            raise ValueError("assembler has unresolved relocs")
        return self._bytes


class RvTarget:
  def __init__(self, rv):
    if not rv.startswith("rv"):
      raise ValueError("invalid rv target spec")
    rv = rv[2:]
    if rv.startswith("32i"):
      self.word_size = 32
      rv = rv[3:]
    elif rv.startswith("64i"):
      self.word_size = 64
      rv = rv[3:]
    else:
      raise ValueError("invalid rv target spec")
    sp = rv.split("+")
    self.ext = [str(x).lower() for x in sp[0]]
    for x in sp[1:]:
      self.ext.append(x.lower())

  def __str__(self):
    return '+'.join([f"rv{self.word_size}i"] + self.ext)


VREG = Ops.CUSTOMI
def vreg(n: int):
  return f"r{n}"

ASM = Ops.CUSTOM

def asm_op(nam, *args, **kw):
  ty = kw.get("dtype", dtypes.void)
  return UOp(ASM, arg=nam, dtype=ty, src=tuple(args))

# TODO: sometimes its better to multiply by a higher amount and then subtract a few times
def gen_mult(op: UOp, amount: int) -> UOp:
  if amount == 0:
    return UOp.const(0, dtype=op.dtype)
  if amount < 0:
    op = gen_mult(op, abs(amount))
    return -op
  if amount == 1:
    return op

  sh = math.floor(math.log2(amount))
  # since the rewriter will rewrite the pattern multiple times, this is fine
  out = (op << sh)
  amount -= 2**sh
  if amount == 1:
    out = out + op
  if amount > 1:
    out = out + op * amount
  return out


# in the "rv" field of the ctx (Renderer)
class RvCtx:
  def __init__(self):
    self.next_reg = 0

  def mkreg(self):
    v = self.next_reg
    self.next_reg += 1
    return v


def emit_const(out, x):
  lui_mask = 0b11111111111111111111000000000000
  imm_mask = 0b11111111111 # only 11 bit because sign ext
  ux = abs(x)
  if ux & imm_mask:
    return asm_op("addi", out, RvRegs.zero, x, dtype=out.dtype)
  if ux & lui_mask == x:
    return asm_op("lui", out, x, dtype=out.dtype)
  if x < 0:
    # TODO: impl this somehow (note that addi sign extends the imm!!)
    raise ValueError("big signed imm not implemented")
  if ux & (lui_mask | imm_mask) != x:
    raise ValueError("unimplemented")
  return asm_op("addi",
                asm_op("lui", out, x & lui_mask, dtype=out.dtype),
                RvRegs.zero, x & imm_mask, dtype=out.dtype)

def imm_fits_in_bits(imm: int, bits: int) -> bool:
  imm = abs(imm)
  return imm < (1 << bits)


def rewrite_shiftadd(ctx: Renderer, op: UOp, d: UOp):
  a = op.src[1].src[1]
  sh = op.src[1].src[2].arg
  b = op.src[2]
  if not post_check_pat_gp_reg(a, b):
    return None

  if sh <= 0: # should be handled elsewhere
    return None

  if sh in (1,2,3):
    return asm_op(f"sh{sh}add", d, a, b)

  return None


def rv_cg(target: RvTarget):
  word_uint = None
  word_sint = None
  if target.word_size == 32:
    word_uint = dtypes.uint32
    word_sint = dtypes.int32
  elif target.word_size == 64:
    word_uint = dtypes.uint64
    word_sint = dtypes.int64
  else:
    assert False

  word_int = (word_uint, word_sint)
  max_word_uint = (dtypes.bool, dtypes.uint8, dtypes.uint16, dtypes.uint32, word_uint)
  max_word_sint = (dtypes.int8, dtypes.int16, dtypes.int32, word_sint)
  max_word_int = tuple(set(max_word_uint) | set(max_word_sint))
  max_word_store = max_word_int

  # we only have 11 bits for unsigned ints because it is sign extended
  imm_n_bits = 11
  shift_n_bits = 5 if target.word_size == 32 else 6

  pat_gp_reg = UPat.any(UPat(VREG, dtype=max_word_store),
                        UPat(ASM),
                        UPat(Ops.ASSIGN, dtype=max_word_store, src=(UPat(VREG), UPat())),
                        UPat(Ops.CONST, arg=0), # zero reg
                        # will take care of these way later:
                        UPat(Ops.RANGE),
                        UPat(Ops.WHERE, dtype=max_word_store),
                        UPat(Ops.DEFINE_GLOBAL),
                        )

  def post_check_pat_gp_reg(ctx, *op) -> bool:
    def check_gp_or_const(x):
      return (x.op == Ops.CONST) or (len(pat_gp_reg.match(x, store={})) > 0 and post_check_pat_gp_reg(ctx, x))

    if len(op) > 1:
      return all([post_check_pat_gp_reg(ctx, x) for x in op])
    op = op[0]
    if not isinstance(op, UOp):
      return post_check_pat_gp_reg(ctx, *op)
    if op.op == ASM:
      return op.arg in all_rv_ops_with_reg_dest
    if op.op == Ops.RANGE or op.op == Ops.WHERE:
      return all([check_gp_or_const(x) for x in op.src])
    return True

  pre_codegen = [
      # rewrite RECIP with FDIV
      (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
      # rewrite MAX to CMPLT + WHERE
      (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),

      (UPat(Ops.MUL, src=(
          UPat.var("x", dtype=max_word_int),
          UPat(Ops.CONST, arg=-1),
          ), name="op"),
      lambda x,op: UOp(Ops.SUB, dtype=op.dtype, src=(
          UOp(Ops.CONST, arg=0, dtype=op.dtype),
          x
          ))),

      (UPat(Ops.MUL, src=(
          UPat.var("x"),
          UPat(Ops.CONST, arg=1),
          )),
      lambda x: x),

      # rewrite mul by constant to shift + adds
      # TODO: do similar for div
      (UPat(Ops.MUL, src=(
          UPat.var("a"), 
          UPat(Ops.CONST, name="b"),
          )),
      lambda a, b: gen_mult(a, b.arg)),

  ]

  # first elt in pair is how many dest registers it has
  #   currently can only be 0 or 1
  codegen = [
      (1, UPat(Ops.LOAD, dtypes.int8, src=(
              UPat(Ops.ADD, src=(
                  pat_gp_reg,
                  UPat(Ops.CONST, dtype=max_word_int),
              ))
          )),
      lambda ctx,op,d: asm_op("lb", d, op.src[0].src[0], op.src[0].src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[0].src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0].src[0]) else None),

      (1, UPat(Ops.LOAD, dtypes.int16, src=(
              UPat(Ops.ADD, src=(
                  pat_gp_reg,
                  UPat(Ops.CONST, dtype=max_word_int),
              ))
          )),
      lambda ctx,op,d: asm_op("lh", d, op.src[0].src[0], op.src[0].src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[0].src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0].src[0]) else None),

      (1, UPat(Ops.LOAD, word_int, src=(
              UPat(Ops.ADD, src=(
                  pat_gp_reg,
                  UPat(Ops.CONST, dtype=max_word_int),
              ))
          )),
      lambda ctx,op,d: asm_op("lw", d, op.src[0].src[0], op.src[0].src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[0].src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0].src[0]) else None),

      (1, UPat(Ops.LOAD, dtypes.uint8, src=(
              UPat(Ops.ADD, src=(
                  pat_gp_reg,
                  UPat(Ops.CONST, dtype=max_word_int),
              ))
          )),
      lambda ctx,op,d: asm_op("lbu", d, op.src[0].src[0], op.src[0].src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[0].src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0].src[0]) else None),

      (1, UPat(Ops.LOAD, dtypes.uint16, src=(
              UPat(Ops.ADD, src=(
                  pat_gp_reg,
                  UPat(Ops.CONST, dtype=max_word_int),
              ))
          )),
      lambda ctx,op,d: asm_op("lhu", d, op.src[0].src[0], op.src[0].src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[0].src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0].src[0]) else None),

      (1, UPat(Ops.STORE, (dtypes.int8,dtypes.uint8), src=(
              UPat(Ops.ADD, src=(
                  pat_gp_reg,
                  UPat(Ops.CONST, dtype=max_word_int),
              )),
              pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("sb", op.src[0].src[0], op.src[1], op.src[0].src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[0].src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, op.src[0].src[0]) else None),

      (1, UPat(Ops.STORE, (dtypes.int16,dtypes.uint16), src=(
              UPat(Ops.ADD, src=(
                  pat_gp_reg,
                  UPat(Ops.CONST, dtype=max_word_int),
              )),
              pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("sh", op.src[0].src[0], op.src[1], op.src[0].src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[0].src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, op.src[0].src[0]) else None),

      (1, UPat(Ops.STORE, word_int, src=(
              UPat(Ops.ADD, src=(
                  pat_gp_reg,
                  UPat(Ops.CONST, dtype=max_word_int),
              )),
              pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("sw", op.src[0].src[0], op.src[1], op.src[0].src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[0].src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, op.src[0].src[0]) else None),

      (1, UPat(Ops.STORE, (dtypes.uint8,dtypes.int8), src=(
              pat_gp_reg,
              pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("sb", op.src[0], op.src[1], 0, dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, op.src[0]) else None),

      (1, UPat(Ops.STORE, (dtypes.uint16,dtypes.int16), src=(
              pat_gp_reg,
              pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("sh", op.src[0], op.src[1], 0, dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, op.src[0]) else None),

      (1, UPat(Ops.STORE, word_int, src=(
              pat_gp_reg,
              pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("sw", op.src[0], op.src[1], 0, dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, op.src[0]) else None),


      # ==== immediate compute ====
      (1, UPat(Ops.ADD, max_word_int, (
          pat_gp_reg,
          UPat(Ops.CONST, dtype=max_word_int),
          )),
      lambda ctx,op,d: asm_op("addi", d, op.src[0], op.src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      (1, UPat(Ops.INDEX, src=(
          pat_gp_reg,
          UPat(Ops.CONST, dtype=max_word_int),
          )),
      lambda ctx,op,d: asm_op("addi", d, op.src[0], op.src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      # TODO: need more checks?
      (1, UPat(Ops.CAST, max_word_int, (
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("addi", d, op.src[0], 0, dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      (1, UPat(Ops.BITCAST, max_word_int, (
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("addi", d, op.src[0], 0, dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      (1, UPat(Ops.XOR, max_word_int, (
          pat_gp_reg,
          UPat(Ops.CONST, dtype=max_word_int),
          )),
      lambda ctx,op,d: asm_op("xori", d, op.src[0], op.src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      (1, UPat(Ops.OR, max_word_int, (
          pat_gp_reg,
          UPat(Ops.CONST, dtype=max_word_int),
          )),
      lambda ctx,op,d: asm_op("ori", d, op.src[0], op.src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      (1, UPat(Ops.AND, max_word_int, (
          pat_gp_reg,
          UPat(Ops.CONST, dtype=max_word_int),
          )),
      lambda ctx,op,d: asm_op("andi", d, op.src[0], op.src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      (1, UPat(Ops.SHL, max_word_int, (
          pat_gp_reg,
          UPat(Ops.CONST, dtype=max_word_int),
          )),
      lambda ctx,op,d: asm_op("slli", d, op.src[0], op.src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      (1, UPat(Ops.SHR, max_word_int, (
          pat_gp_reg,
          UPat(Ops.CONST, dtype=max_word_int),
          )),
      lambda ctx,op,d: asm_op("srli", d, op.src[0], op.src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      (1, UPat(Ops.CMPLT, max_word_sint, (
          pat_gp_reg,
          UPat(Ops.CONST, dtype=max_word_int),
          )),
      lambda ctx,op,d: asm_op("slti", d, op.src[0], op.src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

      (1, UPat(Ops.CMPLT, max_word_uint, (
          pat_gp_reg,
          UPat(Ops.CONST, dtype=max_word_int),
          )),
      lambda ctx,op,d: asm_op("sltiu", d, op.src[0], op.src[1], dtype=op.dtype)
          if imm_fits_in_bits(op.src[1].arg, bits=shift_n_bits) and
             post_check_pat_gp_reg(ctx, d, op.src[0]) else None),

#    # in rv32:
#    #   rd = rs1 s>> (imm &  b11111)
#    # in rv64:
#    #   rd = rs1 s>> (imm & b111111)
#    SRAI  = op(Encodings.I_shift, shift=0b01000, func3=0b101, opc=0b0010011)

      (1, UPat(Ops.ADD, max_word_int, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("add", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.INDEX, src=(
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("add", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.SUB, max_word_int, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("sub", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.AND, max_word_int, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("and", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.OR, max_word_int, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("or", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.XOR, max_word_int, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("xor", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      # TODO: need to generate multiple shifts and subs when we don't know if shift amount might be bigger than [shift_n_bits]
      (1, UPat(Ops.SHL, max_word_int, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("sll", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.SHR, max_word_int, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d,s1: asm_op("srl", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

#    # in rv32:
#    #   rd = rs1 s>> (rs2 &  b11111)
#    # in rv64:
#    #   rd = rs1 s>> (rs2 & b111111)
#    SRA   = op(Encodings.R, func7=0b0100000, func3=0b101, opc=0b0110011)

      (1, UPat(Ops.CMPLT, max_word_sint, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d,s1: asm_op("slt", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.CMPLT, max_word_uint, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d,s1: asm_op("sltu", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

#    # rd = pc + 4
#    # pc += sext(offset)
#    JAL   = op(Encodings.J, opc=0b1101111)
#
#    # t = pc + 4
#    # pc = rs1 + sext(offset)
#    # rd = t
#    JALR  = op(Encodings.I, func3=0b000, opc=0b1100111)
#
#    # if (rs1 == rs2)
#    #   pc += sext(offset)
#    BEQ   = op(Encodings.B, func3=0b000, opc=0b1100011)
#
#    # if (rs1 != rs2)
#    #   pc += sext(offset)
#    BNE   = op(Encodings.B, func3=0b001, opc=0b1100011)
#
#    # if (rs1 s< rs2)
#    #   pc += sext(offset)
#    BLT   = op(Encodings.B, func3=0b100, opc=0b1100011)
#
#    # if (rs1 s>= rs2)
#    #   pc += sext(offset)
#    BGE   = op(Encodings.B, func3=0b101, opc=0b1100011)
#
#    # if (rs1 u< rs2)
#    #   pc += sext(offset)
#    BLTU  = op(Encodings.B, func3=0b110, opc=0b1100011)
#
#    # if (rs1 u>= rs2)
#    #   pc += sext(offset)
#    BGEU  = op(Encodings.B, func3=0b111, opc=0b1100011)

  ]

  codegen = (codegen + [
      # ==== reg-reg compute ====
      (1, UPat(Ops.MUL, max_word_int, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("mul", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

#    # upper part of MUL
#    # rd = ((signed xlen) rs1 * (signed xlen) rs2) s>> xlen
#    MULH  = op(Encodings.R, func7=0b0000001, func3=0b001, opc=0b0110011)
#
#    # upper part of MUL
#    # rd = ((signed xlen) rs1 * (un-signed xlen) rs2) s>> xlen
#    MULHSU= op(Encodings.R, func7=0b0000001, func3=0b010, opc=0b0110011)
#
#    # upper part of MUL
#    # rd = ((un-signed xlen) rs1 * (un-signed xlen) rs2) u>> xlen
#    MULHU = op(Encodings.R, func7=0b0000001, func3=0b011, opc=0b0110011)

      (1, UPat(Ops.IDIV, max_word_sint, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("div", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.IDIV, max_word_uint, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("divu", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.MOD, max_word_sint, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("rem", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

      (1, UPat(Ops.MOD, max_word_uint, (
          pat_gp_reg,
          pat_gp_reg,
          )),
      lambda ctx,op,d: asm_op("remu", d, op.src[0], op.src[1], dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d, op.src) else None),

  ]) if "m" in target.ext else codegen

  codegen = (codegen + [
    (1, UPat(ASM, arg="add", src=(
          pat_gp_reg, # dest
          UPat(ASM, arg="slli", src=(
            pat_gp_reg, # dest
            pat_gp_reg,
            UPat(Ops.CONST),
          )),
          pat_gp_reg,
        )),
      rewrite_shiftadd)
  ]) if "zba" in target.ext else codegen

  new_codegen = []
  for ndest, pat, fn in codegen:
    new_codegen.append((pat_gp_reg.named("d").assign(pat.named("op")), fn))
    if ndest > 0:
      # TODO: can remove?
      def wrp(ctx, fn=fn, **kwargs):
        newreg = UOp(VREG, dtype=kwargs["op"].dtype, arg=vreg(ctx.rv.mkreg()))
        o = fn(ctx=ctx, d=newreg, **kwargs)
        return o
      new_codegen.append((pat.named("op"), wrp))
  codegen = new_codegen

  codegen = codegen + [

      (pat_gp_reg.named("d").assign(
        UPat(Ops.CONST, max_word_int, name="c")),
      lambda ctx,d,s: emit_const(d,s)
          if post_check_pat_gp_reg(ctx, d,s) else None),

      (pat_gp_reg.named("d").assign(
        pat_gp_reg.named("s")),
      lambda ctx,d,s: asm_op("addi", d, s, 0, dtype=op.dtype)
          if post_check_pat_gp_reg(ctx, d,s) else None),

  ]

  if getenv("VIZ"):
    return TrackedPatternMatcher(pre_codegen + codegen)
  else:
    return PatternMatcher(pre_codegen + codegen)

# TODO: remove eventually
from tinygrad.ops import track_rewrites
@track_rewrites()
def full_graph_rewrite(sink, rewr, ctx: Renderer|None):
  from tinygrad.codegen.devectorizer import pm_reduce,gep_pushing,sym,devectorize,load_store_folding,correct_load_store,load_store_indexing,ReduceContext
  sink = graph_rewrite(sink, pm_reduce+gep_pushing, ctx=ReduceContext(), name="remove_reduce")
  sink = graph_rewrite(sink, sym+devectorize+load_store_folding+correct_load_store+load_store_indexing+rewr, ctx=ctx, bottom_up=False, name="lower")
  return sink

# TODO: remove eventually
def debug_flatten(op: UOp) -> tuple[list[str], str]:
  if op.op == Ops.ASSIGN:
    return (debug_flatten(op.src[1])[0], op.src[0].arg)
  elif op.op in [Ops.CONST, VREG]:
    return ([], str(op.arg))
  elif op.op == ASM:
    args = []
    lio = []
    for x in op.src:
      li,v = debug_flatten(x)
      args.append(v)
      lio = lio + li
    x = ", ".join(args)
    return (lio + [f"{op.arg} {x}\n"], "")
  elif op.op == Ops.SINK:
    lio = []
    for x in op.src:
      li,v = debug_flatten(x)
      lio = lio + li
    return (lio, "")
  else:
    return ([], op.__repr__())

def debug(op: UOp) -> str:
  li, x = debug_flatten(op)
  return "".join(li)


class RvRenderer(Renderer):
    device = "RV"
    supports_float4 = False
    has_shared = False
    has_local = False
    global_max = None
    rv = RvCtx()
    target: RvTarget = None

    def __init__(self, target: str = "rv32im+Zba"):
        self.target = RvTarget(target)
        self.graph_rewriter = rv_cg(self.target)
        #self.extra_matcher = self.graph_rewriter

    def render(self, uops: list[UOp]) -> str:
        assert self.target is not None
        print("rendering:")
        for op in uops:
            print("", op)
        return "hahaha no"
