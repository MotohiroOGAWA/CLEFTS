from __future__ import annotations

from .....libs.msentity.msentity import MSDataset

def open_spectrum_window(
    index_text: str,
    dataset: MSDataset | None,
):
    print("=" * 80)
    print("open_spectrum_window called")
    print("index_text =", repr(index_text))

    if dataset is None:
        print("dataset is None")
        return

    try:
        index = int(index_text)
    except ValueError:
        print("invalid index")
        return

    if index < 0:
        print("index is negative")
        return

    print("dataset type =", type(dataset))
    print("dataset length =", len(dataset))
    print("dataset[index] =", dataset[index])