bestScoreDic = {}
bestScoreLookUpDic = {}
wholeRuleDic = {}
nodeSerial = 0
class RuleNode:
    def __init__(self, LHS, RLHS, RRHS):
        global nodeSerial
        self.serial = nodeSerial
        self.LHS = LHS
        self.RLHS = RLHS
        self.RRHS = RRHS
        self.accumuProba = -float('inf')
        self.proba = None
        self.seen = False

        self.cordi = None
        self.tempUnaryNode = None
        self.leftBackNode = None
        self.rightBackNode = None
        self.unaryBackNode = None
        self.treeDicRawID = None
        self.backBackNode = None

        self.parenLHS = ''

        nodeSerial += 1
# =================================================================
def printTree(treeDic):
    # process first Node in raw in TreeDic
    iC = 0
    for rowID, nodeList in treeDic.items():
        # if rowID == 3: break
        if rowID == 0:
            continue
        sumStr = ''
        firstNode = nodeList[0]
        firstParentNode = firstNode.backBackNode
        for treeNode in treeDic[firstParentNode.treeDicRawID]:
            sumStr = sumStr + ' ' * len(str(treeNode.parenLHS))
            if (str(treeNode.LHS).startswith('BIN_') == False) and (str(treeNode.LHS).startswith('NT_') == False):
                sumStr += ' '

            if treeNode == firstParentNode:
                if (str(firstNode.LHS).startswith('BIN_')) or (str(treeNode.LHS).startswith('NT_')):
                    firstNode.parenLHS = sumStr
                else:
                    firstNode.parenLHS = sumStr + firstNode.parenLHS
                break

    for rawNum, nodeList in treeDic.items():
        # if rawNum == 15: break
        sumStr = ""
        for node in nodeList:
            sumStr = sumStr + node.parenLHS
            if (str(node.LHS).startswith('BIN_') == False) and (str(node.LHS).startswith('NT_') == False):
                sumStr = sumStr + ' '
        print(sumStr)
    #
        # print(str().join(str(node.LHS) + ' ' for node in nodeList))
        # print(''.join(str(node.LHS)+'('+str(node.serial)+'<-'+str(node.backBackNode.serial)+')'+" -> " for node in nodeList))
        # print(str().join('->'+str(node.parenLHS)+'<-'for node in nodeList), '\n')

def findMostRightNode(firstNode, terminalSet):
    iC = 0
    while(True):
        # if iC == 5: break
        if firstNode.rightBackNode == None and firstNode.unaryBackNode == None:
            # print("most right Node->",  firstNode.LHS)
            if firstNode.parenLHS == '':
                firstNode.parenLHS = firstNode.LHS + ')'
            else:
                firstNode.parenLHS += ')'
            break

        if firstNode.rightBackNode != None:
            firstNode = firstNode.rightBackNode
        else:
            firstNode = firstNode.unaryBackNode
        iC += 1
def findPath(rootNode, n, terminalSet):
    rootNode.treeDicRawID = 0
    tempNode = RuleNode("", None, None)
    rootNode.backBackNode = tempNode
    nodeQueue = [rootNode]
    treeDic = {}
    for iC in range(n):
        treeDic[iC] = []
    treeDic[0].append(rootNode)
    iTime = 0
    while len(nodeQueue) > 0:
        # while iTime < 1 :
        cNode = nodeQueue.pop(0)
        if cNode.parenLHS == '' and (cNode.LHS in terminalSet): # cNode.ruleStr.find(')') == -1:
            cNode.parenLHS = cNode.LHS

        if (cNode.LHS.startswith('BIN_') == False) and (cNode.LHS.startswith('NT_') == False):
            if (cNode.LHS in terminalSet) == False:
                cNode.parenLHS = '(' + cNode.LHS
                findMostRightNode(cNode, terminalSet)

        # print('000 iTime->', iTime, cNode.LHS, [node.LHS for node in nodeQueue] , '# nodeQueue->',len(nodeQueue))
        unaryBnode, leftBnode, rightBnode = cNode.unaryBackNode, cNode.leftBackNode, cNode.rightBackNode
        if leftBnode != None:
            leftBnode.backBackNode = cNode
            lRawDiff = cNode.cordi[0] - leftBnode.cordi[0]
            lNewTreeRawID = cNode.treeDicRawID + lRawDiff
            leftBnode.treeDicRawID = lNewTreeRawID
            treeDic[lNewTreeRawID].append(leftBnode)

            nodeQueue.append(leftBnode)
            # print('@Left---->', leftBnode.LHS)
        if unaryBnode != None:
            unaryBnode.backBackNode = cNode
            cRawDiff = cNode.cordi[0] - unaryBnode.cordi[0]
            cNewTreeRawID = cNode.treeDicRawID + cRawDiff
            unaryBnode.treeDicRawID = cNewTreeRawID
            treeDic[cNewTreeRawID].append(unaryBnode)

            nodeQueue.append(unaryBnode)
            # print('@Center---->', unaryBnode.LHS)
        if rightBnode != None:
            rightBnode.backBackNode = cNode
            rRawDiff = rightBnode.cordi[0] - cNode.cordi[0]
            rNewTreeRawID = cNode.treeDicRawID + rRawDiff
            rightBnode.treeDicRawID = rNewTreeRawID
            treeDic[rNewTreeRawID].append(rightBnode)

            nodeQueue.append(rightBnode)
            # print('@Right---->', rightBnode.LHS)
        # print('-----------------')
        # for i, nodes in treeDic.items():
        #     # print(i, [(node.ruleStr, 'SN->',node.serial, node.backBackParent.serial ) for node in nodes])
        #     print(i, [(node.LHS, 'SN->',node.serial, node.backBackParent.serial ) for node in nodes])
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        iTime += 1

    # for i, nodes in treeDic.items():
    #     # print(str().join(str(node.LHS)+'('+str(node.serial)+'<-'+str(node.backBackNode.serial)+') ->' for node in nodes  ))
    #     print(str().join(str(node.parenLHS) + ' ' for node in nodes  ))

    return treeDic
def nodeUnaryFun(firstBSnode, cordi):
    if ((firstBSnode.LHS,) in wholeRuleDic) == False:return
    unaryBSList = []
    for node in wholeRuleDic[(firstBSnode.LHS,)]:
        BSnode = bestScoreLookUpDic[cordi][node.LHS, node.RLHS, node.RRHS]
        BSnode.tempUnaryNode = firstBSnode
        unaryBSList.append(BSnode)

    iC = 0
    while (len(unaryBSList) > 0):
    # while(iC < 10):
    #     print(111111, [(node.LHS, node.proba) for node in unaryBSList])
        unaryBSnode = unaryBSList.pop(0)
        tempAccmuProba = unaryBSnode.tempUnaryNode.accumuProba + unaryBSnode.proba
        if tempAccmuProba > unaryBSnode.accumuProba:
            # print(7777, unaryBSnode.LHS,  tempAccmuProba, unaryBSnode.accumuProba)
            unaryBSnode.accumuProba = tempAccmuProba
            unaryBSnode.unaryBackNode = unaryBSnode.tempUnaryNode
            unaryBSnode.leftBackNode, unaryBSnode.rightNode = None, None
            if (unaryBSnode.LHS, ) in wholeRuleDic:
                for node in wholeRuleDic[(unaryBSnode.LHS, )]:
                    BSnode = bestScoreLookUpDic[cordi][node.LHS, node.RLHS, node.RRHS]
                    BSnode.tempUnaryNode = unaryBSnode

                    BSnode.seen = True
                    unaryBSList.append(BSnode)
        iC += 1
def updateUnaryRule(cordi, unaryUpdateList, showOnNot = False):
    iC = 0
    if len (unaryUpdateList) == 0:
        unaryUpdateList = []
        for BSnode in bestScoreDic[cordi]:
            unaryUpdateList.append(BSnode)
    while(len(unaryUpdateList) > 0):
        # if iC == 1 : break
        updateNode = unaryUpdateList.pop(0)

        if updateNode.seen == False:
            nodeUnaryFun(updateNode, cordi)
        iC += 1

def findLeftRightRule(leftTuple, rightTuple,centerTuple, fromRHS):
    # print(88888, leftTuple, rightTuple, centerTuple)
    for leftNode in bestScoreDic[leftTuple]:
        if leftNode.accumuProba == -float('inf'):break
        # if centerTuple == (1, 3):
            # print('>>>>>>>>', leftNode.LHS, leftNode.accumuProba, len(bestScoreDic[rightTuple]))
        for rightNode in bestScoreDic[rightTuple]:
            if rightNode.accumuProba == -float('inf'):break
            if (leftNode.LHS, rightNode.LHS) in wholeRuleDic.keys():
                # print(111111, centerTuple, '<<<---',  (leftNode.LHS, leftNode.accumuProba), (rightNode.LHS, rightNode.accumuProba))
                matchNodeList = wholeRuleDic[(leftNode.LHS, rightNode.LHS)]
                matchBSnodeList = [bestScoreLookUpDic[centerTuple][(node.LHS, node.RLHS, node.RRHS)] for node in matchNodeList]
                for matchBSnode in matchBSnodeList:
                    tempAccumuProba = leftNode.accumuProba + rightNode.accumuProba + matchBSnode.proba
                    if tempAccumuProba > matchBSnode.accumuProba:
                        matchBSnode.accumuProba = tempAccumuProba
                        matchBSnode.leftBackNode = leftNode
                        matchBSnode.rightBackNode = rightNode

def printBestScore(n):
    for iC, (cordi, nodeList) in enumerate(bestScoreDic.items()):
        print(cordi, [(node.LHS, round(node.accumuProba, 2) )for node in nodeList if node.accumuProba > -float('inf')])

def initializeCell(rhsRulesDic, n):
    # iC = 0
    def ruleDicFun(cordi):
        returnRuleList = []
        lookUPDic = {}
        for iC, (rhsKey, ruleList) in enumerate(rhsRulesDic.items()):
            wholeRuleDic[rhsKey] = []
            # if iC == 10: break
            for oneRuleList in ruleList:
                if len(rhsKey) == 1:
                    newRuleNode = RuleNode(oneRuleList[0], oneRuleList[1][0], None)
                    newRuleNode2 = RuleNode(oneRuleList[0], oneRuleList[1][0], None)
                    lookUPDic[(oneRuleList[0], oneRuleList[1][0], None)] = newRuleNode2
                else:
                    newRuleNode = RuleNode(oneRuleList[0], oneRuleList[1][0], oneRuleList[1][1])
                    newRuleNode2 = RuleNode(oneRuleList[0], oneRuleList[1][0], oneRuleList[1][1])
                    lookUPDic[(oneRuleList[0], oneRuleList[1][0], oneRuleList[1][1])] = newRuleNode2
                newRuleNode.proba = oneRuleList[2]
                newRuleNode2.proba = oneRuleList[2]
                newRuleNode.cordi, newRuleNode2.cordi = cordi, cordi
                wholeRuleDic[rhsKey].append(newRuleNode)
                returnRuleList.append(newRuleNode2)
        return returnRuleList, lookUPDic

    for i in range(n):
        # if i == 1:break
        for j in range(i+1, n+ 1):
            bestScoreDic[(i,j)], bestScoreLookUpDic[(i, j)] = ruleDicFun((i, j))
# =======================================================================================
# =======================================================================================





