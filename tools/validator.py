class Validator:
    def __init__(self, folder_path, amplitude_threshold=0.01):
        self.folder_path = folder_path
        self.amplitude_threshold = amplitude_threshold
        self.main_sign = self.get_main_sign()

    def get_main_sign(self):
        pos, neg = 0, 0
        with open(f"{self.folder_path}/tmp/signal.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                if (
                    line.strip() == ""
                    or float(line.strip().split(",")[0]) < self.amplitude_threshold
                ):
                    continue
                if float(line.strip().split(",")[-1]) != -1:
                    pos += 1
                else:
                    neg += 1
        return 1 if pos > neg else -1

    def validated_indices(self):
        with open(f"{self.folder_path}/tmp/signal.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.strip() == "" or float(line.strip().split(",")[0]) < 1:
                    continue
                if float(line.strip().split(",")[-1]) == self.main_sign:
                    yield lines.index(line)
