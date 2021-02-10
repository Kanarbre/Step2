def bits(n, size):
    output = [0] * size
    for i in range(size):
        output[i] = (n >> i) & 1
    return output


def ddt_rows(ddt, in_size, out_size):
    out = {}
    for a in range(1 << in_size):
        out[a] = set()
        for b in range(1 << out_size):
            if ddt[a, b] != 0:
                out[a].add(b)
    return out


def ddt_cols(ddt, in_size, out_size):
    out = {}
    for b in range(1 << out_size):
        out[b] = set()
        for a in range(1 << in_size):
            if ddt[a, b] != 0:
                out[b].add(a)
    return out


def parse_space(s):
    assert len(s) % 8 == 0
    c = 0
    temp = ""
    for x in s:
        if c == 8:
            temp = temp + " "
            c = 1
            temp = temp + x
        else:
            temp = temp + x
            c += 1
    return temp
