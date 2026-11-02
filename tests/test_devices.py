import devices


def test_auto_with_two_gpus():
    pool, warns = devices.resolve("auto", ["cuda:0", "cuda:1"])
    assert pool == ["cuda:0", "cuda:1"] and warns == []


def test_auto_without_gpu_is_cpu():
    assert devices.resolve("auto", [])[0] == ["cpu"]


def test_cpu_explicit():
    assert devices.resolve("cpu", ["cuda:0"])[0] == ["cpu"]


def test_list_parsed():
    pool, _ = devices.resolve("cuda:0,cuda:1", ["cuda:0", "cuda:1"])
    assert pool == ["cuda:0", "cuda:1"]


def test_missing_gpu_falls_back_with_warning():
    pool, warns = devices.resolve("cuda:3", ["cuda:0"])
    assert pool == ["cuda:0"] and warns and "cuda:3" in warns[0]


def test_missing_gpu_no_cuda_falls_back_to_cpu():
    pool, warns = devices.resolve("cuda:0", [])
    assert pool == ["cpu"] and warns


def test_round_robin_assignment():
    got = devices.assign(["a", "b", "c"], ["cuda:0", "cuda:1"], {})
    assert [got[k] for k in "abc"] == ["cuda:0", "cuda:1", "cuda:0"]


def test_override_wins_when_available():
    got = devices.assign(["a", "b"], ["cuda:0", "cuda:1"], {"a": "cuda:1"})
    assert got["a"] == "cuda:1"


def test_override_unavailable_ignored():
    got = devices.assign(["a"], ["cuda:0"], {"a": "cuda:5"})
    assert got["a"] == "cuda:0"
