class str_buffer:

    def __init__(self):
        self.current_pointer = 0
        self.on_hold_list = []
        self.on_hold_length = 0
        self.length = 1000

    def append(self, s):
        if len(s) + self.on_hold_length < self.length:




