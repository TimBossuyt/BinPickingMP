class HashTable:
    def __init__(self):
        self.table = dict()

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, value):
        self.table[key] = value

    def items(self):
        """
        Returns an iterable view of the hash table's items (key-value pairs).
        """
        return self.table.items()

    def keys(self):
        return self.table.keys()

    def get(self, key, default=None):
        return self.table[key] if key in self.table else default