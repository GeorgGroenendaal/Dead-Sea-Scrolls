def greeting(msg: list[str]) -> None:
    match msg:
        case ["greeting", word]:
            print(f"Hello there {word}")
        case ["goodbye", word]:
            print(f"See you later {word}")
        case [_, word]:
            print(f"Unknown greeting {word}")
        case []:
            raise ValueError("Please provide a greeting")
        case _:
            raise ValueError("Don't know how to solve this")
