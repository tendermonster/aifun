import os
import time


class Logger:
    def __init__(self, log_to_console=True):
        self._timestamp = int(time.time())
        self.log_name = "session" + "_" + str(self._timestamp) + ".txt"
        self.root = "logs"
        self.log_to_console = log_to_console
        self.file_path = os.path.join(self.root, self.log_name)

    def get_timestamp(self):
        return str(self._timestamp)

    def log(self, message):
        full = time.ctime() + ": " + message
        with open(os.path.join(self.file_path), "a") as f:
            f.write(full + "\n")
            f.close()
        if self.log_to_console:
            print(full)
