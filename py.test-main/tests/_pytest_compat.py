"""Zero-dependency pytest compatibility shim for standalone execution of tests/."""
from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import logging
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _RaisesContext:
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.exc_type}")
        if not issubclass(exc_type, self.exc_type):
            return False
        self.value = exc_val
        if self.match is not None and not re.search(self.match, str(exc_val)):
            raise AssertionError(f"Regex {self.match!r} did not match {str(exc_val)!r}")
        return True


def raises(exc_type, match=None):
    return _RaisesContext(exc_type, match=match)


def fixture(*args, **kwargs):
    def _dec(fn):
        fn._is_fixture = True
        fn._autouse = kwargs.get("autouse", False)
        return fn
    if len(args) == 1 and callable(args[0]) and not kwargs:
        args[0]._is_fixture = True
        args[0]._autouse = False
        return args[0]
    return _dec


class _Mark:
    def parametrize(self, argnames, argvalues, **kwargs):
        def _dec(fn):
            if not hasattr(fn, "_parametrize"):
                fn._parametrize = []
            fn._parametrize.append((argnames, list(argvalues)))
            if not hasattr(fn, "pytestmark"):
                fn.pytestmark = []
            fn.pytestmark.append(types.SimpleNamespace(name="parametrize", args=(argnames, list(argvalues)), kwargs=kwargs))
            return fn
        return _dec

    def __getattr__(self, name):
        def _decorator(*a, **kw):
            if len(a) == 1 and callable(a[0]) and not kw:
                return a[0]
            return lambda f: f
        return _decorator


mark = _Mark()


class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=..., raising=True):
        if value is ...:
            value = name
            mod_path, attr = target.rsplit(".", 1)
            target = importlib.import_module(mod_path)
            name = attr
        old = getattr(target, name, ...)
        if old is ... and raising:
            raise AttributeError(f"{target} has no attribute {name}")
        self._undo.append(("attr", target, name, old))
        setattr(target, name, value)

    def delattr(self, target, name=..., raising=True):
        if name is ...:
            mod_path, name = target.rsplit(".", 1)
            target = importlib.import_module(mod_path)
        old = getattr(target, name, ...)
        if old is ...:
            if raising:
                raise AttributeError(f"{target} has no attribute {name}")
            return
        self._undo.append(("attr", target, name, old))
        delattr(target, name)

    def setitem(self, dic, name, value):
        old = dic.get(name, ...)
        self._undo.append(("item", dic, name, old))
        dic[name] = value

    def delitem(self, dic, name, raising=True):
        if name not in dic:
            if raising:
                raise KeyError(name)
            return
        old = dic[name]
        self._undo.append(("item", dic, name, old))
        del dic[name]

    def setenv(self, name, value, prepend=None):
        old = os.environ.get(name, ...)
        self._undo.append(("env", os.environ, name, old))
        if prepend and name in os.environ:
            value = f"{value}{prepend}{os.environ[name]}"
        os.environ[name] = str(value)

    def delenv(self, name, raising=True):
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            return
        old = os.environ[name]
        self._undo.append(("env", os.environ, name, old))
        del os.environ[name]

    def chdir(self, path):
        old = os.getcwd()
        self._undo.append(("chdir", None, None, old))
        os.chdir(path)

    def undo(self):
        for kind, target, name, old in reversed(self._undo):
            if kind == "attr":
                if old is ...:
                    try:
                        delattr(target, name)
                    except AttributeError:
                        pass
                else:
                    setattr(target, name, old)
            elif kind in ("item", "env"):
                if old is ...:
                    target.pop(name, None)
                else:
                    target[name] = old
            elif kind == "chdir":
                os.chdir(old)
        self._undo.clear()


class _CapLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class CapLog:
    def __init__(self):
        self.handler = _CapLogHandler()
        self._root = logging.getLogger()
        self._old_level = self._root.level
        self._root.addHandler(self.handler)

    @property
    def records(self):
        return self.handler.records

    @property
    def text(self):
        return "\n".join(f"{r.levelname} {r.name}: {r.getMessage()}" for r in self.records)

    @contextlib.contextmanager
    def at_level(self, level, logger=None):
        target = logging.getLogger(logger) if logger else self._root
        old_lvl = target.level
        target.setLevel(level)
        target.addHandler(self.handler)
        try:
            yield
        finally:
            target.setLevel(old_lvl)
            target.removeHandler(self.handler)

    def cleanup(self):
        self._root.removeHandler(self.handler)
        self._root.setLevel(self._old_level)


class CapSys:
    def __init__(self):
        self._out = io.StringIO()
        self._err = io.StringIO()
        self._old_out = sys.stdout
        self._old_err = sys.stderr
        sys.stdout = self._out
        sys.stderr = self._err

    def readouterr(self):
        out = self._out.getvalue()
        err = self._err.getvalue()
        self._out.seek(0)
        self._out.truncate(0)
        self._err.seek(0)
        self._err.truncate(0)
        return types.SimpleNamespace(out=out, err=err)

    def cleanup(self):
        sys.stdout = self._old_out
        sys.stderr = self._old_err


def fail(msg: str = ""):
    raise AssertionError(msg)


def skip(msg: str = ""):
    return None


def run_module_tests(mod) -> int:
    fixtures = {
        name: obj for name, obj in inspect.getmembers(mod, inspect.isfunction)
        if getattr(obj, "_is_fixture", False)
    }
    autouse = [f for f in fixtures.values() if getattr(f, "_autouse", False)]
    passed = 0
    failed = 0

    def _invoke(fn, fix_map, tmp_dir, mp, cl, cs):
        sig = inspect.signature(fn)
        kwargs = {}
        gens = []
        for pname in sig.parameters:
            if pname == "tmp_path":
                kwargs[pname] = Path(tempfile.mkdtemp(dir=tmp_dir))
            elif pname == "monkeypatch":
                kwargs[pname] = mp
            elif pname == "caplog":
                kwargs[pname] = cl
            elif pname == "capsys":
                kwargs[pname] = cs
            elif pname in fix_map:
                res = _invoke(fix_map[pname], fix_map, tmp_dir, mp, cl, cs)
                if inspect.isgenerator(res):
                    kwargs[pname] = next(res)
                    gens.append(res)
                else:
                    kwargs[pname] = res
        try:
            return fn(**kwargs)
        finally:
            for g in reversed(gens):
                try:
                    next(g)
                except StopIteration:
                    pass

    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(obj)
            res = unittest.TextTestRunner(stream=open(os.devnull, "w")).run(suite)
            passed += res.testsRun - len(res.failures) - len(res.errors)
            failed += len(res.failures) + len(res.errors)

    targets = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_") and not getattr(obj, "_is_fixture", False):
            targets.append((name, obj))
    for cname, cls in inspect.getmembers(mod, inspect.isclass):
        if cname.startswith("Test") and not issubclass(cls, unittest.TestCase):
            inst = cls()
            for mname, mobj in inspect.getmembers(inst, inspect.ismethod):
                if mname.startswith("test_"):
                    targets.append((f"{cname}.{mname}", mobj))

    for tname, fn in targets:
        param_specs = getattr(fn, "_parametrize", [])
        call_sets = [{}]
        for argnames, argvalues in param_specs:
            names = [x.strip() for x in argnames.split(",")] if isinstance(argnames, str) else list(argnames)
            new_sets = []
            for cs_dict in call_sets:
                for val in argvalues:
                    ncs = dict(cs_dict)
                    if len(names) == 1:
                        ncs[names[0]] = val
                    else:
                        for k, v in zip(names, val):
                            ncs[k] = v
                    new_sets.append(ncs)
            call_sets = new_sets

        for cs_dict in call_sets:
            with tempfile.TemporaryDirectory() as tmp_dir:
                mp = MonkeyPatch()
                cl = CapLog()
                cs = CapSys()
                gens = []
                try:
                    for af in autouse:
                        r = _invoke(af, fixtures, tmp_dir, mp, cl, cs)
                        if inspect.isgenerator(r):
                            next(r)
                            gens.append(r)
                    local_fix = dict(fixtures)
                    for k, v in cs_dict.items():
                        local_fix[k] = (lambda val=v: val)
                    _invoke(fn, local_fix, tmp_dir, mp, cl, cs)
                    passed += 1
                except Exception as exc:
                    failed += 1
                    sys.__stderr__.write(f"FAIL {tname}: {type(exc).__name__}: {exc}\n")
                finally:
                    for g in reversed(gens):
                        try:
                            next(g)
                        except StopIteration:
                            pass
                    cs.cleanup()
                    cl.cleanup()
                    mp.undo()

    print(f"PASS: {passed} passed, {failed} failed")
    return 1 if failed else 0
