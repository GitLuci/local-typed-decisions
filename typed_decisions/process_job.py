"""Windows lifetime guard for owned servers; no PID lookup or external adoption.

The non-inherited Job Object handle belongs to the API process. Windows kills
assigned children when that process exits, including os._exit/forced termination.
Assignment follows Popen immediately; the small launch-to-assignment window is
not covered. Non-Windows callers retain explicit close()/service-manager cleanup.
"""
import ctypes as ct
from ctypes import wintypes as wt
import os


class BasicLimits(ct.Structure):
    _fields_ = [("process_time", ct.c_int64), ("job_time", ct.c_int64), ("flags", wt.DWORD),
        ("min_ws", ct.c_size_t), ("max_ws", ct.c_size_t), ("active", wt.DWORD),
        ("affinity", ct.c_size_t), ("priority", wt.DWORD), ("scheduling", wt.DWORD)]


class ExtendedLimits(ct.Structure):
    _fields_ = [("basic", BasicLimits), ("io", ct.c_uint64 * 6),
        ("process_memory", ct.c_size_t), ("job_memory", ct.c_size_t),
        ("peak_process", ct.c_size_t), ("peak_job", ct.c_size_t)]


def kernel_api():
    api = ct.WinDLL("kernel32", use_last_error=True)
    for name, args, result in (
        ("CreateJobObjectW", [ct.c_void_p, wt.LPCWSTR], wt.HANDLE),
        ("SetInformationJobObject", [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD], wt.BOOL),
        ("AssignProcessToJobObject", [wt.HANDLE, wt.HANDLE], wt.BOOL),
        ("CloseHandle", [wt.HANDLE], wt.BOOL),
    ):
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    return api


class ProcessJob:
    def __init__(self, *, api=None):
        self.api, self.handle = None, None
        if os.name != "nt" and api is None:
            return
        self.api = api or kernel_api()
        self.handle = self.api.CreateJobObjectW(None, None)  # unnamed, non-inheritable
        if not self.handle:
            raise OSError("Cannot create owned-server Job Object")
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ct.byref(limits), ct.sizeof(limits)):
            self.close()
            raise OSError("Cannot set owned-server lifetime limit")

    def attach(self, process):
        if self.handle is not None:
            # CPython's actual process HANDLE, avoiding PID reuse races.
            if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
                raise OSError("Cannot guard owned-server lifetime")

    def close(self):
        if self.handle is not None:
            if not self.api.CloseHandle(self.handle):
                raise OSError("Cannot close owned-server Job Object")
            self.handle = None
