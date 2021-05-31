import sys
import math
from collections import namedtuple, defaultdict
from itertools import chain, product

START_SYM = 'ROOT'


class GrammarRule(namedtuple('Rule', ['lhs', 'rhs', 'log_prob'])):
    """A named tuple that represents a PCFG grammar rule.

    Each GrammarRule has three fields: lhs, rhs, log_prob

    Parameters
    ----------
    lhs : str
        A string that represents the left-hand-side symbol of the grammar rule.
    rhs : tuple of str
        A tuple that represents the right-hand-side symbols the grammar rule.
    log_prob : float
        The log probability of this rule.
    """

    def __repr__(self):
        return '{} -> {} [{}]'.format(
            self.lhs, ' '.join(self.rhs), self.log_prob)


def read_rules(grammar_filename):
    """Read PCFG grammar rules from grammar file

    The grammar file is a tab-separated file of three columns:
    probability, left-hand-side, right-hand-side.
    probability is a float number between 0 and 1. left-hand-side is a
    string token for a non-terminal symbol in the PCFG. right-hand-side
    is a space-delimited field for one or more  terminal and non-terminal
    tokens. For example::

        1	ROOT	EXPR
        0.333333	EXPR	EXPR + TERM

    Parameters
    ----------
    grammar_filename : str
        path to PCFG grammar file

    Returns
    -------
    set of GrammarRule
    """
    rules = set()
    with open(grammar_filename) as f:
        for rule in f.readlines():
            rule = rule.strip()
            log_prob, lhs, rhs = rule.split('\t')
            rhs = tuple(rhs.split(' '))
            assert rhs and rhs[0], rule
            rules.add(GrammarRule(lhs, rhs, math.log(float(log_prob))))
    return rules


class Grammar:
    """PCFG Grammar class."""

    def __init__(self, rules):
        """Construct a Grammar object from a set of rules.

        Parameters
        ----------
        rules : set of GrammarRule
            The set of grammar rules of this PCFG.
        """
        self.rules = rules

        self._rhs_rules = defaultdict(list)
        self._rhs_unary_rules = defaultdict(list)

        self._nonterm = set(rule.lhs for rule in rules)
        self._term = set(token for rhs in chain(rule.rhs for rule in rules)
                         for token in rhs if token not in self._nonterm)

        for rule in rules:
            _, rhs, _ = rule
            self._rhs_rules[rhs].append(rule)

        for rhs_rules in self._rhs_rules.values():
            rhs_rules.sort(key=lambda r: r.log_prob, reverse=True)

        self._is_cnf = all(len(rule.rhs) == 1
                           or (len(rule.rhs) == 2
                               and all(s in self._nonterm for s in rule.rhs))
                           for rule in self.rules)

    def from_rhs(self, rhs):
        """Look up rules that produce rhs

        Parameters
        ----------
        rhs : tuple of str
            The tuple that represents the rhs.

        Returns
        -------
        list of GrammarRules with matching rhs, ordered by their
        log probabilities in decreasing order.
        """
        return self._rhs_rules[rhs]

    def __repr__(self):
        summary = 'Grammar(Rules: {}, Term: {}, Non-term: {})\n'.format(
            len(self.rules), len(self.terminal), len(self.nonterminal)
        )
        summary += '\n'.join(sorted(self.rules))
        return summary

    @property
    def terminal(self):
        """Terminal tokens in this grammar."""
        return self._term

    @property
    def nonterminal(self):
        """Non-terminal tokens in this grammar."""
        return self._nonterm

    def get_cnf(self):
        """Convert PCFG to CNF and return it as a new grammar object."""
        nonterm = set(self.nonterminal)
        term = set(self.terminal)

        rules = list(self.rules)
        cnf = set()

        # STEP 1: eliminate nonsolitary terminals
        for i in range(len(rules)):
            rule = rules[i]
            lhs, rhs, log_prob = rule
            if len(rhs) > 1:
                rhs_list = list(rhs)
                for j in range(len(rhs_list)):
                    x = rhs_list[j]
                    if x in term:  # found nonsolitary terminal
                        new_nonterm = 'NT_{}'.format(x)
                        new_nonterm_rule = GrammarRule(new_nonterm, (x,), 0.0)

                        if new_nonterm not in nonterm:
                            nonterm.add(new_nonterm)
                            cnf.add(new_nonterm_rule)
                        else:
                            assert new_nonterm_rule in cnf
                        rhs_list[j] = new_nonterm
                rhs = tuple(rhs_list)
            rules[i] = GrammarRule(lhs, rhs, log_prob)

        # STEP 2: eliminate rhs with more than 2 nonterminals
        for i in range(len(rules)):
            rule = rules[i]
            lhs, rhs, log_prob = rule
            if len(rhs) > 2:
                assert all(x in nonterm for x in rhs), rule
                current_lhs = lhs
                for j in range(len(rhs) - 2):
                    new_nonterm = 'BIN_"{}"_{}'.format(
                        '{}->{}'.format(lhs, ','.join(rhs)), str(j))
                    assert new_nonterm not in nonterm, rule
                    nonterm.add(new_nonterm)
                    cnf.add(
                        GrammarRule(current_lhs,
                                    (rhs[j], new_nonterm),
                                    log_prob if j == 0 else 0.0))
                    current_lhs = new_nonterm
                cnf.add(GrammarRule(current_lhs, (rhs[-2], rhs[-1]), 0.0))
            else:
                cnf.add(rule)

        return Grammar(cnf)

    def parse(self, line):
        """Parse a sentence with the current grammar.

        The grammar object must be in the Chomsky normal form.

        Parameters
        ----------
        line : str
            Space-delimited tokens of a sentence.
        """
        # BEGIN_YOUR_CODE
        #
        # # UNARY HANDLING
        # split line to list of words
        words = line.split()
        n = len(words)

        if n == 0:
            return;

        bestScore = [[dict() for _ in range(n + 1)]
                     for i in range(n)]

        # make only unary grammer
        unary_after_grammer = [orig for orig in self.rules if len(orig.rhs) == 1]

        # make only original grammer
        after_grammer = [orig.lhs for orig in self.rules]
        ori_grammer = []
        for lhs in after_grammer:
            if str(lhs).startswith('NT_') == False and str(lhs).startswith('BIN_') == False:
                ori_grammer.append(lhs)
        ori_grammer = set(ori_grammer)


        # add unary func
        def add_unary(child_A, i, j, loop_check):
            if child_A not in loop_check:

                for B_A in unary_after_grammer:
                    if B_A.rhs[0] == child_A:
                        unary_max_score = B_A.log_prob + bestScore[i][j][child_A][1]
                        if B_A.lhs in bestScore[i][j]:
                            unary_current_score = bestScore[i][j][B_A.lhs][1]
                            if max(unary_current_score, unary_max_score) == unary_current_score:
                                current_lhs = bestScore[i][j][B_A.lhs][0]
                            else:
                                current_lhs = child_A
                            bestScore[i][j][B_A.lhs] = ([current_lhs, max(unary_current_score, unary_max_score),bestScore[i][j][B_A.lhs][2]])
                        else:
                            bestScore[i][j][B_A.lhs] = ([child_A, unary_max_score, i])
                        loop_check.append(child_A)
                        add_unary(B_A.lhs, i, j, loop_check)


        # Fill terminal rules
        for i in range(1, n + 1):
            word_t = (words[i - 1],)
            from_rhs_w = Grammar.from_rhs(self, word_t)

            bestScore[i - 1][i][words[i - 1]] = ['',-9999999999999999999,i-1]
            for B_word in from_rhs_w:
                bestScore[i - 1][i][B_word.lhs] = [words[i - 1], B_word.log_prob]

                # add unary rule for the first

                loop_check = []
                add_unary(B_word.lhs, i - 1, i, loop_check)

        # after the second line
        for l in range(2, n + 1):
            for i in range(0, n - l + 1):
                j = i + l
                tmp_max = {}
                for k in range(i + 1, j):
                    left_l = bestScore[i][k].keys()
                    right_l = bestScore[k][j].keys()
                    for left in left_l:
                        for right in right_l:
                            rhs_pattern = (left,right)
                            # pattern exist in the grammar
                            from_rhs_pattern = Grammar.from_rhs(self, rhs_pattern)
                            if len(from_rhs_pattern) != 0:
                                for rule_pattern in from_rhs_pattern:
                                    sum_score = rule_pattern.log_prob + bestScore[i][k][left][1] + \
                                                bestScore[k][j][right][1]

                                    if rule_pattern.lhs not in tmp_max or max(sum_score, tmp_max[rule_pattern.lhs][1]) == sum_score:
                                        tmp_max[rule_pattern.lhs] = [rule_pattern.rhs,sum_score,k]

                for pair_lhs, pair_items in tmp_max.items():
                    bestScore[i][j][pair_lhs] = [pair_items[0], pair_items[1], pair_items[2]]


                # Add unary rules
                if len(bestScore[i][j]) != 0:
                    add_list = [A for A in bestScore[i][j].keys()]
                    for add_ary_target in add_list:
                        loop_check = []
                        add_unary(add_ary_target, i, j, loop_check)

        ###

        def print_target(self, i, j, target,l):
            tree_right = []
            tree_unary = []
            # add target
            if target in words or target in ori_grammer:
                route_l[l].append(target)
                route_check_l[l].append(target)
            # find next target
            if target not in words:
                target_item = bestScore[i][j][target]


                # next_target is unary
                if type(target_item[0]) == str:
                    if len(target_item) != 2:
                        if target_item[2] != i:
                            print_target(self, target_item[2], j, target_item[0],i)
                        else:
                            print_target(self, i, j, target_item[0],i)
                    else:
                        print_target(self, i, j, target_item[0], i)
                # next_target is tree
                elif type(target_item[0]) == tuple:
                    k = target_item[2]

                    route_l[l].append(['TREE'])
                    route_check_l[l].append(['TREE'])
                    tree_right.append([k, j, target_item[0][1]])
                    # recurring for left
                    print_target(self, i, k, target_item[0][0],i)
                    # move to right part
                    right_k = tree_right[-1][0]
                    right_i = tree_right[-1][1]
                    right_target = tree_right[-1][2]
                    tree_right.pop()
                    print_target(self, right_k, right_i, right_target,right_k)

        def print_result():
            global is_no_Root
            is_no_Root = True
            check_cell = bestScore[0][n]
            if START_SYM in check_cell and check_cell[START_SYM][1] != -9999999999999999999:
                is_no_Root = False
                print_target(self, 0, n, START_SYM,0)


        route_l = [[] for i in range(n)]
        route_check_l = [[] for i in range(n)]
        print_result()

        def route_print():
            p_str = ''
            p_len_l = []
            p_len = 0
            for p_word in range(n):
                p_line = route_l[p_word]
                if len(p_len_l) != 0:
                    for space in range(p_len_l[-1]):
                        print(' ', end='')
                    p_len = p_len_l[-1]
                    p_len_l.pop(-1)
                for num, p_result in enumerate(p_line):
                    if type(p_result) == list:
                        p_len_l.append(p_len)
                    elif num != len(p_line) - 1:
                        p_str = '(' + p_result + ' '
                        p_len = p_len + len(p_str)
                        print(p_str, end='')
                    elif num == len(p_line) - 1:
                        c_bracket = 0
                        target_line = p_word + 1
                        for i_check, line in enumerate(route_check_l[p_word::-1]):
                            target_line = target_line - 1
                            for j_check, count in enumerate(line[-2::-1]):
                                if str(count).endswith('_BRACKET') == False and type(count) != list:
                                    route_check_l[target_line][len(line) - j_check - 2] = count + '_BRACKET'
                                    c_bracket = c_bracket + 1
                                    continue
                                elif type(count) == list:
                                    if str(count[0]).endswith('_BRACKET') == False:
                                        route_check_l[target_line][len(line) - j_check - 2] = [count[0] + '_BRACKET']
                                        break
                            else:
                                continue
                            break
                        p_str = p_result
                        print(p_str, end='')
                        for blanket in range(c_bracket):
                            print(')', end='')
                        print('')

        if is_no_Root == False:
            route_print()
            score = bestScore[0][n][START_SYM]
            print(str(score[1]))
        else:
            print('NONE')

        # print('1行おわりー！！！！！')
    # END_YOUR_CODE
