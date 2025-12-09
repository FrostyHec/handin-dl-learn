

class AU:
    @staticmethod
    def getN(args,n):
        if len(args) != n:
            raise TypeError(f"Args mismatch! Expected: {n} parameters.")
        return args