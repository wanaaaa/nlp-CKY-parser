from functionClass import *
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
        # print("========", self._rhs_rules)
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
        # print(11111111111, rhs, '<<<<<-------')
        return self._rhs_rules[rhs]

    def __repr__(self):
        summary = 'Grammar(Rules: {}, Term: {}, Non-term: {})\n'.format(
            len(self.rules), len(self.terminal), len(self.nonterminal)
        )
        print("===>>", summary, "<<====")
        print("<<<<<", sorted(self.rules))
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
        # grammar = Grammar(self.rules)
        cnfGra = self.get_cnf()
        rhsRulesDic = cnfGra._rhs_rules
        # self._rhs_unary_rules = defaultdict(list)
        unaryRuleDic = self._rhs_unary_rules
        # fromRhs = self.from_rhs(('3',))
        fromRHS = self.from_rhs
        lineList = line.strip().split()
        # print(Grammar.from_rhs(self, ('EXPR', 'NT_}')))
        n = len(lineList)
        if n == 0:
            print('NONE')
            return

        initializeCell(rhsRulesDic, n)

        if len(lineList) > 0:
            for iC, word in enumerate(lineList):
                # if iC == 1:break
                # print("<<<", (iC, iC + 1),  word, ">>>>>")
                termiNode = RuleNode(word, None, None)
                termiNode.accumuProba, termiNode.proba = float('inf'), float('inf')
                termiNode.cordi = (iC, iC + 1)
                unaryUpdateList = []
                for iiC, ruleNode in enumerate(wholeRuleDic[(word,)]):
                    # if iiC == 1:break
                    # print("   1111", iiC, ruleNode)
                    nodeInBScoreDic = bestScoreLookUpDic[(iC, iC+ 1)][(ruleNode.LHS, ruleNode.RLHS, ruleNode.RRHS)]
                    nodeInBScoreDic.accumuProba = nodeInBScoreDic.proba
                    nodeInBScoreDic.unaryBackNode = termiNode
                    unaryUpdateList.append(nodeInBScoreDic)
                updateUnaryRule((iC, iC + 1), unaryUpdateList)
                bestScoreDic[(iC, iC + 1)].sort(key = lambda node: node.accumuProba, reverse=True)

        for l in range(2, n+1):
            # if l == 3:break
            # print('@@@@@ l->', l)
            for i in range(0, n - l + 1):
                # if i == 1:break
                # print('   @@@@ i', i)
                j = i + l
                for k in range(i+1, j-1+ 1):
                    # print("<<<  ", '(i,j)->','(',i,",",j,')',' (i,k)->','(', i,",",k,")", ' (k+1,j)->',"(",k,",",j,")", sep="")
                    findLeftRightRule((i,k), (k,j), (i,j), fromRHS)
                bestScoreDic[(i, j)].sort(key=lambda node: node.accumuProba, reverse=True)
                updateUnaryRule((i, j), [])
                bestScoreDic[(i, j)].sort(key=lambda node: node.accumuProba, reverse=True)

        rootNode = RuleNode('NONE', None, None)
        for node in bestScoreDic[0,n]:
            if node.LHS == 'ROOT':
                rootNode = node
        if rootNode.accumuProba == - float('inf'):
            print('NONE')
        else:
            terminalSet = self.terminal
            rootNode = rootNode
            treeDic = findPath(rootNode, n, terminalSet)
            printTree(treeDic)
            print(rootNode.accumuProba)

# ========================================
