from primitive import AesLike
from itertools import product as itp
import time

shift_rows = [0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12]

mixcol_equiv = [
    [0, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 1],
]

mixcol_origin = [
    [1, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 1],
]


class Skinny(AesLike):
    """ Gurobi Model for Skinny-128 differential trails. """

    def __init__(self, nb_rounds, sbox_file, mixcol="equiv"):

        # Multiple choices of MixColumns models
        if mixcol == "equiv":
            self.mixcol = mixcol_equiv
        else:
            self.mixcol = mixcol_origin
        AesLike.__init__(self, 128, nb_rounds, sbox_file)

    def linear_layer(self, x_in, x_out):
        x_in = [
            x_in[(8 * shift_rows[i // 8]) + (i % 8)] for i in range(128)
        ]  # variables after ShiftRows and before MixColumns

        for col, bit in itp(range(4), range(8)):
            bit_list = [x_in[(32 * row) + (8 * col) + bit] for row in range(4)] + [
                x_out[(32 * row) + (8 * col) + bit] for row in range(4)
            ]

            self.add_bin_matrix_constr(
                self.mixcol, bit_list, 0, mode="integer",
            )


if __name__ == "__main__":
    """
    Launch a differential trail search on n rounds of Skinny-128
    """
    n = 9

    file = open("truncated_" + str(n) + "rounds.txt")
    T = file.readlines()
    file.close()

    G = []
    for i in range(len(T)):
        if i % 2 == 1:
            G.append(T[i])

    for i in range(len(G)):
        G[i] = G[i][5:-5]
        temp = ""
        for j in range(len(G[i])):
            if G[i][j] == ',':
                pass
            else:
                temp += G[i][j]

        G[i] = temp
        assert (len(G[i]) == 16 * n)

    file = open("results.txt", "a")
    file.write("Starting time : " + str(time.time())+"\n")
    file.close()

    skinny128 = Skinny(n, "skinny_sbox4.pkl")
    skinny128.model.setParam("LogToConsole", 0)
    skinny128.model.setParam("OutputFlag", 1)
    skinny128.model.setParam("LogFile", "log.txt")

    output = skinny128.differential_trail_search(G[0])
    file = open("results.txt", "a")
    file.write(output + "\nFinish time : " + str(time.time())+"\n")
    file.close()




