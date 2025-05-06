"""
This module provides utilities for generating and manipulating text arrays, 
including random character generation, target text setting, and frame updates. 
It also includes a function to simulate a dynamic size value.
"""

import string
import numpy as np
import numpy.typing as npt

text: npt.NDArray[np.character] = np.array([])
target: npt.NDArray[np.character] = np.array([])
x = 0.0


def increment(char: bytes) -> bytes:
    """
    Increment a character by one over all possible characters in the set of 
    digits, uppercase letters, and lowercase letters. Wraps around when 
    reaching the end of each group.

    Args:
        char (bytes): A single character as a byte.

    Returns:
        bytes: The incremented character as a byte.
    """
    if char == b'9':
        return b'A'
    if char == b'Z':
        return b'a'
    if char == b'z':
        return b'0'

    return bytes([char[0] + 1])


def random_chars(length: int) -> npt.NDArray[np.character]:
    """
    Generate a random array of characters consisting of letters and digits.

    Args:
        length (int): The length of the random character array.

    Returns:
        npt.NDArray[np.character]: An array of random characters.
    """
    return np.random.choice(
        list(string.ascii_letters + string.digits),
        size=length,
    ).astype(np.character)


def random_char() -> str:
    """
    Generate a single random character from the set of letters and digits.

    Returns:
        str: A random character.
    """
    return np.random.choice(list(string.ascii_letters + string.digits))


def set_target(new_target: str):
    """
    Set the target text and adjust the global `text` array to match the target's length.
    Random characters are used to fill any gaps, and spaces in the target are preserved.

    Args:
        new_target (str): The new target string.
    """
    global target
    global text

    target = np.array(list(new_target), dtype=np.character)

    if len(text) == 0:
        text = np.array(
            list("The Third Dimension"),
            dtype=np.character
        )  # loop

    if len(text) > len(target):
        text = text[:len(target)]

    if len(text) < len(target):
        diff = len(target) - len(text)
        text = np.concatenate((text, random_chars(diff)))

    text[text == b' '] = random_chars(len(text[text == b' ']))
    text[target == b' '] = b' '


def get_frame(n=1) -> str:
    """
    Incrementally update the global `text` array to match the target text 
    and return the resulting string. If `n` is greater than 1, the function 
    recursively updates the frame `n` times.

    Args:
        n (int): The number of frames to process. Defaults to 1.

    Returns:
        str: The updated text as a string.
    """
    global target
    global text

    if np.any(text != target):
        f = np.vectorize(increment)
        text[text != target] = f(text[text != target])

    if n != 1:
        return get_frame(n-1)

    return b''.join(text.tolist()).decode("utf-8")


def get_size() -> float:
    """
    Simulate a dynamic size value that oscillates over time.

    Returns:
        float: The computed size value.
    """
    global x
    x += 0.1
    return np.sin(x) * 0.25 + 1.75
