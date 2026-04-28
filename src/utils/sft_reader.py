class SFTReader:
    """
    Класс для чтения sft файлов на указанный device.
    """

    def __init__(self, sft_path: str, device: str = "cpu"):
        self.sft_path = sft_path
        self.device = device

    def read_sft(self):
        """
        Загружает .sft/.safetensors напрямую на целевой device.
        Это убирает лишний проход CPU -> GPU copy -> CPU.
        """
        from safetensors.torch import load_file

        return load_file(self.sft_path, device=self.device)
