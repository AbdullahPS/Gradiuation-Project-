class ImageStone:

    def __init__(self, src=0,original=0, threshed=0, holesFilled=0, cleaned=0, adjusted=0, filtered=0, dilated=0,gray=0):
        self.src = src
        self.original=original
        self.threshed = threshed
        self.holesFilled = holesFilled
        self.cleaned = cleaned
        self.adjusted = adjusted
        self.filtered = filtered
        self.dilated = dilated
        self.gray=gray

    def myfunc(self):
        print("Hello my name is " + self.src)
