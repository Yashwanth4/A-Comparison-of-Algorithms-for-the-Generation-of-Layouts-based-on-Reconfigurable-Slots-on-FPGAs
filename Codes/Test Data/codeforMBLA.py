import sys
import random as rand
import sympy
import os

class Block():
    def __init__(self, ID, area):
        self.ID = ID
        self.area = area

    def __repr__(self):
        return "block{}".format(self.ID)

    def __str__(self):
        return "ID\t: {} \nArea\t: {}".format(self.ID, self.area)


def createBlock(i, amin, amax):
    a = rand.randint(amin, amax)
    if (sympy.isprime(a) and (a > W)): a = a - 1
    return Block(i, a)


def createBlockSet(sizeFPGA):
    sizeBlocks = 0
    Blocks = []
    i = 0
    while True:
        Blocks.append(createBlock(i + 1, 1, 15))
        sizeBlocks = sizeBlocks + Blocks[i].area
        if not (sizeBlocks < sizeFPGA):
            Blocks.pop()
            sizeBlocks = sum(eachBlk.area for eachBlk in Blocks)
            break
        i = i + 1
    ss = sizeFPGA - sizeBlocks
    if (sympy.isprime(ss) and (ss > W)): ss = ss - 1
    Blocks.append(createBlock(i + 1, ss, ss))
    return Blocks


def main(file_num):
    global W
    W = rand.randint(15, 25)
    H = W
    sizeFPGA = W * H
    area_util = rand.uniform(0.40, 0.9)
    SetBlock = createBlockSet(int(sizeFPGA * area_util))
    file = "MBLA_Data/MBLA_{}.txt".format(file_num)


    with open(file, "w") as f:
        f.write("Outline: {} {}".format(W, H))
        for blk in SetBlock:
            f.write("\nBlock{} {}".format(blk.ID, blk.area))
        # for i, blk in enumerate(SetBlock):
        #     if i == 0:
        #         f.write("Block{} {}".format(blk.ID, blk.area))
        #     else:
        #         f.write("\nBlock{} {}".format(blk.ID, blk.area))


if __name__ == "__main__":
    for i in range(60):
        main(i)
