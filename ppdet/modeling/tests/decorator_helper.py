import paddle.fluid as fluid

__all__ = ['prog_scope']


def prog_scope():
    def __impl__(fn):
        def __fn__(*args, **kwargs):
            prog = fluid.Program()
            startup_prog = fluid.Program()
            scope = fluid.core.Scope()
            with fluid.scope_guard(scope):
                with fluid.program_guard(prog, startup_prog):
                    with fluid.unique_name.guard():
                        fn(*args, **kwargs)

        return __fn__

    return __impl__
