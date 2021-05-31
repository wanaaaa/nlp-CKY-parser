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

        Returns
        -------
        log_probs : {(start, end): {lhs: log_prob}}
            Log probability for each span with both start and end
            being inclusive, for the lhs of each matched rules.
        backpointer : {(start, end): {lhs: (partition, [rule])}}
            Backpointer for each span with both start and end
            being inclusive, for the lhs of each matched rules.
            partition represents the length of the first rhs symbol.
            The list of rules are the rules with which to produce the
            lhs.
        length : number of tokens in the input sentence.
        """
        assert self._is_cnf, "The current grammar isn't in CNF."
        line = line.strip()
        if not line:
            return {}, {}, 0
        tokens = line.split(' ')
        length = len(tokens)
        log_probs = defaultdict(defaultdict)
        backpointer = defaultdict(defaultdict)
        # BEGIN_YOUR_CODE

        # END_YOUR_CODE

        self.print_parse_result(line, log_probs, backpointer, length)
        return log_probs, backpointer, length

    def find_parents(self, symbol):
        """Find the transitive closure of all parents of a symbol.

        Returns
        -------
        list of tuple of form (symbol, path) where the symbol is one
        of the input symbol's parent and path is a list of rules to
        traverse from the input symbol to its parent.
        """
        return self._find_parents_helper(symbol, [], set())

    def _find_parents_helper(self, symbol, path, seen):
        results = [(symbol, tuple(path))]
        # BEGIN_YOUR_CODE

        # END_YOUR_CODE
        return results

    def _fill_unary_parents(self, log_probs, backpointer,
                            token, log_prob, start, end):
        # BEGIN_YOUR_CODE

        # END_YOUR_CODE
        backpointer[(start, end)][lhs_par] = (end - start + 1, path[-1:])

    def _fill_unary_terminal(self, log_probs, backpointer, token, pos):
        # BEGIN_YOUR_CODE

        # END_YOUR_CODE
        self._fill_unary_parents(log_probs, backpointer,
                                 lhs, log_prob, pos, pos)

    def _fill_binary_nonterminal(self, log_probs, backpointer,
                                 start, span_len, partition):
        # BEGIN_YOUR_CODE

        # END_YOUR_CODE
        self._fill_unary_parents(log_probs, backpointer,
                                 lhs, log_prob, start, end)

    def print_parse_result(self, line, log_probs, backpointer, length):
        """Pretty-print the parsing results.

        Parameters
        ----------
        line : str
            Space-delimited tokens of a sentence.
        log_probs : {(start, end): {lhs: log_prob}}
            Log probability for each matched rules.
        backpointer : {(start, end): {lhs: (partition, [rule])}}
            Backpointer for each span with both start and end
            being inclusive, for the lhs of each matched rules.
            partition represents the length of the first rhs symbol.
            The list of rules are the rules with which to produce the
            lhs.
        length : number of tokens in the input sentence.

        See also
        --------
        self.parse(line)
        """
        line = line.strip()
        if (0, length - 1) in log_probs and \
                START_SYM in log_probs[(0, length - 1)]:
            print(
                self._format_parse(START_SYM, backpointer, 0, length - 1, 0))
            print(log_probs[(0, length - 1)][START_SYM])
        else:
            # print('Parsing failed "{}"'.format(line))
            print('NONE')

    def _format_parse(self, symbol, backpointer, start, end, indent):
        if symbol in self.terminal:
            assert start == end, (start, end, symbol)
            return symbol
        else:
            partition, path = backpointer[(start, end)][symbol]
            # skip unary non-terminal transformation symbols
            if symbol.startswith('NT_'):
                return self._format_parse(
                    path[-1].rhs[0], backpointer, start, end, indent)

            # skip binarization transformation symbols
            if not symbol.startswith('BIN_'):
                indent = indent + 2 + len(symbol)

            if start + partition - 1 == end:  # unary expansion
                children_str = self._format_parse(
                    path[-1].rhs[0], backpointer, start, end, indent)
            else:  # binary non-terminal rules
                delimiter = '\n' + ' ' * indent
                children_str = delimiter.join([
                    self._format_parse(
                        path[-1].rhs[0], backpointer,
                        start, start + partition - 1, indent),
                    self._format_parse(
                        path[-1].rhs[1], backpointer,
                        start + partition, end, indent)])
            if not symbol.startswith('BIN_'):
                return '({} {})'.format(symbol, children_str)
            else:
                return children_str


def main():
    assert len(sys.argv) == 3, "Usage: python3 parse.py foo.gr foo.sen"
    grammar_filename = sys.argv[1]
    filename = sys.argv[2]

    rules = read_rules(grammar_filename)
    grammar = Grammar(rules)
    cnf_grammar = grammar.get_cnf()

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        log_probs, backpointer, length = cnf_grammar.parse(line)


if __name__ == '__main__':
    main()