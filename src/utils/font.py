_CHAR_MAP = {
    "Alef": ")",
    "Ayin": "(",
    "Bet": "b",
    "Dalet": "d",
    "Gimel": "g",
    "He": "x",
    "Het": "h",
    "Kaf": "k",
    "Kaf-final": "\\",
    "Lamed": "l",
    "Mem": "{",
    "Mem-medial": "m",
    "Nun-final": "}",
    "Nun-medial": "n",
    "Pe": "p",
    "Pe-final": "v",
    "Qof": "q",
    "Resh": "r",
    "Samekh": "s",
    "Shin": "$",
    "Taw": "t",
    "Tet": "+",
    "Tsadi-final": "j",
    "Tsadi-medial": "c",
    "Waw": "w",
    "Yod": "y",
    "Zayin": "z",
}


def text_to_font(inp: str) -> str:
    out = ""
    lines = inp.splitlines()

    for line in lines:
        characters = line.strip().split(" ")

        for character in characters:
            if character not in _CHAR_MAP:
                raise ValueError(f"Unknown character map {character}")
            out += _CHAR_MAP[character]

        out += "\n"

    return out
