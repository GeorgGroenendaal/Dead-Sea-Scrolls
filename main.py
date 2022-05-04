from src.hello import greeting


if __name__ == "__main__":
    greeting(["greeting", "world"])
    greeting(["goodbye", "world"])

    try:
        greeting(["hello", "world", "there"])
    except Exception as e:
        print(e)
