def test_api_stub():
    from bh_speech import transcribe
    try:
        transcribe("nepostojeci.wav")
        assert False, "mora baciti NotImplementedError"
    except NotImplementedError:
        pass
