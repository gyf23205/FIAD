def init(hypers):
    global hd1
    global hd2
    global rep
    global T
    # hd1, hd2, rep = 1024, 256, 1024
    hd1, hd2, rep, T = hypers[0], hypers[1], hypers[2], hypers[3]