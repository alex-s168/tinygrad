from tinygrad.device import Compiled, Compiler, MallocAllocator
from tinygrad.renderer.rv import RvRenderer
from tinygrad.runtime.ops_cpu import CPUProgram

class RVCompiler(Compiler):
    def __init__(self):
        super().__init__(None)

    def compile(self, src:str) -> bytes:
        print("tried compile", src)
        return []

    def disassemble(self, lib:bytes):
        assert False

class RVDevice(Compiled):
    def __init__(self, device:str):
        # TODO: don't hardcode
        super().__init__(device, MallocAllocator, RvRenderer("rv32im+Zba"),
                      RVCompiler(), CPUProgram)
