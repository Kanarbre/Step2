from gurobipy import *
import pickle
import utilities
from itertools import product as itp


class Primitive:
    """ Gurobi Model of a cryptographic primitive for differential properties. """

    def __init__(self, in_size, out_size):

        # Dictionary of S-boxes modelings used in this primitive
        self.sbox_modelings = {}

        # Input and output sizes
        self.in_size = in_size
        self.out_size = out_size

        # Input and output Gurobi variables
        self.in_var = {}
        self.out_var = {}

        # Gurobi Model
        self.model = Model()

        # Dictionaries of pb-DDT binary variables
        self.Q64 = None
        self.Q48 = None
        self.Q40 = None
        self.Q32 = None
        self.Q28 = None
        self.Q24 = None
        self.Q20 = None
        self.Q16 = None
        self.Q12 = None
        self.Q8 = None
        self.Q6 = None
        self.Q4 = None
        self.Q2 = None

    def add_sbox_modeling(self, file_name, other_name=None):
        """
        Loads the pickle file file_name of an S-box modeling.
        The pickle file should contain a set of lists representing
        inequalities.
        If ineq is such a list, the input coefficients are first,
        then come the output coefficients and finally comes the constant.
        The inequality is then:
        sum(input[i] * ineq[i]) + sum(output[i] * ineq[i + len(input)])
        + ineq[len(input) + len(output)] >= 0
        """
        with open(file_name, "rb") as f:
            if other_name is None:
                other_name = file_name
            (in_size, out_size, ddt, ineq0, ineq64, ineq48, ineq40, ineq32, ineq28, ineq24, ineq20, ineq16, ineq12,
             ineq8, ineq6, ineq4, ineq2) = pickle.load(f)
            self.sbox_modelings[other_name] = (
                utilities.ddt_rows(ddt, in_size, out_size),
                utilities.ddt_cols(ddt, in_size, out_size),
                ineq0,
                ineq64,
                ineq48,
                ineq40,
                ineq32,
                ineq28,
                ineq24,
                ineq20,
                ineq16,
                ineq12,
                ineq8,
                ineq6,
                ineq4,
                ineq2
            )

    def add_sbox_constr(self, sbox_name, a, b, r, i):
        """ 
        Adds constraints for one S-box registered in
        add_sbox_modelings[sbox_name].
        a is a list of input variables,
        b is a list of output variables,
        """
        M = 100
        n = len(a)
        m = len(b)
        (_, _, ineq0, ineq64, ineq48, ineq40, ineq32, ineq28, ineq24, ineq20, ineq16, ineq12, ineq8, ineq6, ineq4, ineq2) = self.sbox_modelings[sbox_name]
        for ineq in ineq64:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q64[r, i])
                >= 0
            )
        for ineq in ineq48:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q48[r, i])
                >= 0
            )
        for ineq in ineq40:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q40[r, i])
                >= 0
            )
        for ineq in ineq32:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q32[r, i])
                >= 0
            )

        for ineq in ineq28:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q28[r, i])
                >= 0
            )

        for ineq in ineq24:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q24[r, i])
                >= 0
            )

        for ineq in ineq20:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q20[r, i])
                >= 0
            )

        for ineq in ineq16:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q16[r, i])
                >= 0
            )

        for ineq in ineq12:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q12[r, i])
                >= 0
            )

        for ineq in ineq8:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q8[r, i])
                >= 0
            )

        for ineq in ineq6:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q6[r, i])
                >= 0
            )

        for ineq in ineq4:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q4[r, i])
                >= 0
            )

        for ineq in ineq2:
            assert len(ineq) == n + m + 1
            self.model.addConstr(
                quicksum(ineq[i] * a[i] for i in range(n))
                + quicksum(ineq[i + n] * b[i] for i in range(m))
                + ineq[n + m]
                + M * (1-self.Q2[r, i])
                >= 0
            )

    def add_xor_constr(self, variables, offset=0, mode="binary"):
        """
        If mode = "binary", adds the $2^{n-1}$ constraints modeling
        the XOR constraint x[0] ^ ... ^ x[n-1] = offset
        where x is variables.
        If mode = "integer", models the same XOR constraint with a dummy
        integer variable t with x[0] + ... + x[n-1] = 2 * t + offset.
        """
        x = variables
        n = len(x)

        if mode == "binary" or mode == "both":
            for i in range(1 << n):
                bit_list = utilities.bits(i, n)
                if sum(bit_list) % 2 == (1 - offset):
                    constraint = quicksum(
                        x[j] if bit_list[j] == 0 else 1 - x[j] for j in range(n)
                    )
                    self.model.addConstr(constraint >= 1)
        if mode == "integer" or mode == "both":
            offset = offset % 2

            t = self.model.addVar(
                name="dummy_xor", lb=0, ub=(n // 2) + (n % 2), vtype=GRB.INTEGER
            )
            self.model.addConstr(quicksum(x) == (2 * t) + offset)

    def add_bin_matrix_constr(self, matrix, x, b, mode="binary"):
        """
        Adds constraints given by matrix * x = b
        where x is a list of GRB.BINARY variables
        and b is a constant given as an integer
        """
        y = utilities.bits(b, len(matrix))

        for i in range(len(matrix)):
            row = matrix[i]
            assert len(row) == len(x)
            variables = [x[j] for j in range(len(x)) if row[j] != 0]
            self.add_xor_constr(variables, offset=y[i], mode=mode)


class AesLike(Primitive):
    """
    Gurobi Model of an AES-like primitive.
    ie, when there is only one permutation S-Box
    and one linear layer
    """

    def __init__(self, state_size, nb_rounds, sbox_file):

        Primitive.__init__(self, state_size, state_size)

        self.nb_rounds = nb_rounds
        self.sbox_name = sbox_file

        with open(sbox_file, "rb") as f:
            (in_nibble_size, out_nibble_size, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = pickle.load(f)

        assert in_nibble_size == out_nibble_size
        nibble_size = in_nibble_size

        assert state_size % nibble_size == 0
        nb_nibbles = state_size // nibble_size

        self.add_sbox_modeling(sbox_file)

        self.state_size = state_size    # equal to 128
        self.nibble_size = nibble_size
        self.nb_nibbles = nb_nibbles

        in_sbox = {}    # in_sbox for sbox input
        out_sbox = {}   # out_sbox for sbox output
        Q64 = {}
        Q48 = {}
        Q40 = {}
        Q32 = {}
        Q28 = {}
        Q24 = {}
        Q20 = {}
        Q16 = {}
        Q12 = {}
        Q8 = {}
        Q6 = {}
        Q4 = {}
        Q2 = {}

        for i, j in itp(range(nb_rounds), range(state_size)):
            in_sbox[i, j] = self.model.addVar(
                name="in_sbox_({}, {})".format(i, j), vtype=GRB.BINARY
            )
        for i, j in itp(range(nb_rounds), range(state_size)):
            out_sbox[i, j] = self.model.addVar(
                name="out_sbox_({}, {})".format(i, j), vtype=GRB.BINARY
            )
        for r, i in itp(range(nb_rounds), range(nb_nibbles)):
            Q64[r, i] = self.model.addVar(
                name="Q64_({}, {})".format(r, i), vtype=GRB.BINARY, obj=2.00,
            )
            Q48[r, i] = self.model.addVar(
                name="Q48_({}, {})".format(r, i), vtype=GRB.BINARY, obj=2.42,
            )
            Q40[r, i] = self.model.addVar(
                name="Q40_({}, {})".format(r, i), vtype=GRB.BINARY, obj=2.68,
            )
            Q32[r, i] = self.model.addVar(
                name="Q32_({}, {})".format(r, i), vtype=GRB.BINARY, obj=3.00,
            )
            Q28[r, i] = self.model.addVar(
                name="Q28_({}, {})".format(r, i), vtype=GRB.BINARY, obj=3.20,
            )
            Q24[r, i] = self.model.addVar(
                name="Q24_({}, {})".format(r, i), vtype=GRB.BINARY, obj=3.42,
            )
            Q20[r, i] = self.model.addVar(
                name="Q20_({}, {})".format(r, i), vtype=GRB.BINARY, obj=3.68,
            )
            Q16[r, i] = self.model.addVar(
                name="Q16_({}, {})".format(r, i), vtype=GRB.BINARY, obj=4.00,
            )
            Q12[r, i] = self.model.addVar(
                name="Q12_({}, {})".format(r, i), vtype=GRB.BINARY, obj=4.42,
            )
            Q8[r, i] = self.model.addVar(
                name="Q8_({}, {})".format(r, i), vtype=GRB.BINARY, obj=5.00,
            )
            Q6[r, i] = self.model.addVar(
                name="Q6_({}, {})".format(r, i), vtype=GRB.BINARY, obj=5.42,
            )
            Q4[r, i] = self.model.addVar(
                name="Q4_({}, {})".format(r, i), vtype=GRB.BINARY, obj=6.00,
            )
            Q2[r, i] = self.model.addVar(
                name="Q2_({}, {})".format(r, i), vtype=GRB.BINARY, obj=7.00,
            )

        self.Q64 = Q64
        self.Q48 = Q48
        self.Q40 = Q40
        self.Q32 = Q32
        self.Q28 = Q28
        self.Q24 = Q24
        self.Q20 = Q20
        self.Q16 = Q16
        self.Q12 = Q12
        self.Q8 = Q8
        self.Q6 = Q6
        self.Q4 = Q4
        self.Q2 = Q2

        for i in range(state_size):
            self.in_var[i] = in_sbox[0, i]
            self.out_var[i] = out_sbox[nb_rounds - 1, i]

        for i in range(nb_rounds):
            self.subcell(
                [in_sbox[i, j] for j in range(state_size)],
                [out_sbox[i, j] for j in range(state_size)],
                i,
            )

        for i in range(nb_rounds - 1):
            self.linear_layer(
                [out_sbox[i, j] for j in range(state_size)],
                [in_sbox[i + 1, j] for j in range(state_size)],
            )

        self.in_sbox = in_sbox
        self.out_sbox = out_sbox

    def subcell(self, in_sbox, out_sbox, r):
        n = self.nb_nibbles
        d = self.nibble_size
        for nibble in range(n):
            a = [in_sbox[(d * nibble) + i] for i in range(d)]
            b = [out_sbox[(d * nibble) + i] for i in range(d)]
            self.add_sbox_constr(self.sbox_name, a, b, r, nibble)

    def linear_layer(self, x_in, x_out):
        """
        Adds linear layer constraints for one round
        """
        raise NotImplementedError

    def differential_trail_search(self, active_sboxes):
        """
        Computes a differential trail with the highest possible probability given a class of
        truncated differential characteristics
        """
        local_constraints = []
        y = dict()
        idx = 0

        for r, i in itp(range(self.nb_rounds), range(self.nb_nibbles)):
            y[r, i] = self.model.addVar(
                name="active_({}, {})".format(r, i), vtype=GRB.BINARY
            )
            bits = [
                self.in_sbox[r, (self.nibble_size * i) + j]
                for j in range(self.nibble_size)
            ]+[self.out_sbox[r, (self.nibble_size * i) + j] for j in range(self.nibble_size)]

            lc = self.model.addGenConstrOr(y[r, i], bits)
            local_constraints.append(lc)

            lc = self.model.addConstr(y[r, i] == quicksum(
                [self.Q64[r, i], self.Q48[r, i], self.Q40[r, i], self.Q32[r, i], self.Q28[r, i],
                 self.Q24[r, i], self.Q20[r, i], self.Q16[r, i], self.Q12[r, i],
                 self.Q8[r, i], self.Q6[r, i], self.Q4[r, i], self.Q2[r, i]
                 ])
            )
            local_constraints.append(lc)

            lc = self.model.addConstr(y[r, i] == int(active_sboxes[idx]))
            local_constraints.append(lc)

            idx += 1

        # We fix at least one active input cell
        lc = self.model.addConstr(
            quicksum(y[0, i] for i in range(self.nb_nibbles)) >= 1
        )
        local_constraints.append(lc)

        # Additional constraint to indicate the minimum number of active S-boxes
        idx = {2: 2.0, 3: 5.0, 4: 8.0, 5: 12.0, 6: 16.0, 7: 26.0, 8: 36.0,
               9: 41.0, 10: 46.0, 11: 51.0, 12: 55.0, 13: 58.0, 14: 61.0}
        lc = self.model.addConstr(
            quicksum(y[r, i] for r, i in itp(range(self.nb_rounds), range(self.nb_nibbles))) >= idx[self.nb_rounds]
        )
        local_constraints.append(lc)

        self.model.optimize()

        # Parsing results
        output = ""
        file = open("SKINNY128_DDT.txt", "r")
        diff = file.readlines()
        file.close()
        probability_index = []
        diff = diff[2:]
        for i in range(len(diff)):
            temp = diff[i][23:]
            assert temp[1] == '/' or temp[2] == '/'
            if temp[1] == '/':
                probability_index.append(int(temp[0]))
            else:
                probability_index.append(int(temp[:2]))
            diff[i] = diff[i][:8]+diff[i][13:21]

        output += "Skinny-128, %s rounds\n" % self.nb_rounds
        output += "Truncated differential characteristics : %s\n" % active_sboxes

        all_status = {2: "OPTIMAL", 6: "CUTOFF", 9: "TIME_LIMIT"}
        if self.model.status in all_status and self.model.SolCount > 0:
            output += "Model status : %s\n" % all_status[self.model.status]

            total = 0
            for r, i in itp(range(self.nb_rounds), range(self.nb_nibbles)):
                total += y[r, i].x
            output += "Best probability found : 2^(-%s) | Number of active S-boxes : %s\n" % (self.model.getObjective().getValue(), total)

            for r in range(self.nb_rounds):
                temp = ""
                temp2 = ""
                transition_probability = ""
                for i in range(self.nb_nibbles):
                    if y[r, i].x >= 0.5:
                        current_diff = ""
                        a = [self.in_sbox[r, (self.nibble_size * i) + j].x for j in range(self.nibble_size)]
                        for x in a:
                            if x >= 0.5:
                                current_diff = current_diff + "1"
                            else:
                                current_diff = current_diff + "0"

                        temp = temp + "{:02x}".format(int(current_diff, 2))

                        current_diff2 = ""
                        b = [self.out_sbox[r, (self.nibble_size * i) + j].x for j in range(self.nibble_size)]

                        for x in b:
                            if x >= 0.5:
                                current_diff = current_diff + "1"
                                current_diff2 = current_diff2 + "1"
                            else:
                                current_diff = current_diff + "0"
                                current_diff2 = current_diff2 + "0"

                        temp2 = temp2 + "{:02x}".format(int(current_diff2, 2))
                        transition_probability = transition_probability + str(probability_index[diff.index(current_diff)]) + "/256."

                    else:
                        temp = temp + "00"
                        temp2 = temp2 + "00"

                output += "Round : %s | Before SB : %s | After SB : %s  | Probability : %s\n" \
                          % (r + 1, utilities.parse_space(temp), utilities.parse_space(temp2), transition_probability)

            output += "-----------------\n"

        else:
            output += "Failure, model status %s\n" % self.model.status

        # Removing local constraints and variables
        for lc in local_constraints:
            self.model.remove(lc)
        for r, i in itp(range(self.nb_rounds), range(self.nb_nibbles)):
            self.model.remove(y[r, i])

        return output
