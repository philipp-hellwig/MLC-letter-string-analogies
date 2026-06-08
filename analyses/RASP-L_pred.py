import np_rasp as rsp
import matplotlib.pyplot as plt

# Input string
s = "abcdefghijklmnopqrstuvwxyz|ghi>fhi|bcd"

# Convert chars to ascii ints
def tn(txt):
    return [ord(t) for t in txt]

## One layer
# Build selectors (QK circuits). These are probably general to all tasks
ppslct = rsp.select(k=tn(s), q=tn('|')*len(s), pred=rsp.equals, causal=False)
gtslct = rsp.select(k=tn(s), q=tn('>')*len(s), pred=rsp.equals, causal=False)
smslct = rsp.select(k=tn(s), q=tn(s),          pred=rsp.equals, causal=False) 

# Find the index of the first character after each separator
x1_ind = rsp.aggr(ppslct, v=rsp.indices(s)+1, reduction='min')
x2_ind = rsp.aggr(gtslct, v=rsp.indices(s)+1, reduction='min')
qy_ind = rsp.aggr(ppslct, v=rsp.indices(s)+1, reduction='max')

# Get first occurrence of each element of string
# Example output for s = "abcdefghij|ghi>fhi|bcd"
# [ 0  1  2  3  4  5  6  7  8  9 10  6  7  8 14  5  7  8 10  1  2  3]
fstind = rsp.aggr(smslct, v=rsp.indices(s),   reduction='min')

## Next layer
# Locates x1, x2, and qy in alphabet
x1slct = rsp.select(k=rsp.indices(fstind), q=x1_ind, pred=rsp.equals, causal=False)
x2slct = rsp.select(k=rsp.indices(fstind), q=x2_ind, pred=rsp.equals, causal=False)
qyslct = rsp.select(k=rsp.indices(fstind), q=qy_ind, pred=rsp.equals, causal=False)

# To route qyind to next level
id     = rsp.select(k=rsp.indices(s), q=rsp.indices(s), pred=rsp.equals, causal=True)

x1_fst = rsp.aggr(x1slct, v=fstind, reduction='mean')
x2_fst = rsp.aggr(x2slct, v=fstind, reduction='mean')
qy_fst = rsp.aggr(qyslct, v=fstind, reduction='mean')
qy_ind

# Compute transformation of first element of query
# Corresponds to MLP operation
out_ind = x2_fst-x1_fst + qy_fst

## Next layer
# Note that the comparison is not exact because Rasp does not yet support decoder and cross-att
# Build component attention matrices id, o_slct, cpslct and combine using logical operations
id     = rsp.select(k=rsp.indices(s), q=rsp.indices(s), pred=rsp.equals, causal=True)
o_slct = rsp.select(k=rsp.indices(s), q=out_ind,        pred=rsp.equals, causal=True)
cpslct = rsp.select(k=rsp.indices(s), q=qy_ind,         pred=rsp.gt,     causal=True)

# Compare to last layer of decoder
o_slct = (o_slct | cpslct ) & id

# Non-zero elements of out1 correspond to 
out = rsp.aggr(o_slct, v=tn(s), default=0, reduction='mean')

# Viewing output:
# Outputs ['a', 'c', 'd']
out_view = [chr(c) for c in out if c != 0]
print(f"Nonzero output: {out_view}")



fig, axs = plt.subplots(1, 3, figsize = (15, 5))
axs[0].imshow(ppslct)
axs[0].set_title("Pipe selector (ppslct)")
axs[1].imshow(gtslct)
axs[1].set_title("Arrow selector (gtslct)")
axs[2].imshow(smslct)
axs[2].set_title("Same-as selector (smslct)")

# axs[3].imshow(interv
for ax in axs:
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s)
# plt.savefig("rasp_layer1.png")
# plt.show()

fig, axs = plt.subplots(1, 3, figsize = (15, 5))
axs[0].imshow(x1slct)
axs[0].set_title("Example 1 selector (x1slct)")
axs[1].imshow(x2slct)
axs[1].set_title("Example 2 selector (x2slct)")
axs[2].imshow(qyslct)
axs[2].set_title("Query selector (qyslct)")
for ax in axs:
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s)
# plt.savefig("rasp_layer2.png")

fig, axs = plt.subplots(1, 1, figsize = (5, 5))
axs.imshow(o_slct)
axs.set_title("Output selector (o_slct)")
for ax in [axs]:
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s)
# plt.savefig("rasp_layer3.png")
plt.show()

fig, axs = plt.subplots(1, 1, figsize = (5, 5))
axs.imshow(smslct)
axs.set_title("Same-as selector (smslct)")
for ax in [axs]:
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s)
plt.savefig("rasp_full.png")
plt.show()










# # decoder layer?
# dec_qy = '>'
# dcslct = rsp.select(k=s, q=dec_qy, pred=rsp.equals, causal=False)
# print(dcslct)
# print(np.array([out_ind]))
# # print(np.append(out_ind,0))
# out1 = rsp.aggr(dcslct[0], out_ind, 0, reduction='mean') #hack to fix

# print(f"out1: {out1}")

# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(dcslct)
# # axs[1].imshow(x2slct)
# # axs[2].imshow(qyslct)
# for ax in axs:
#     ax.set_xticks(range(len(s)))
#     ax.set_xticklabels(s)
#     ax.set_yticks(range(len(dec_qy)))
#     ax.set_yticklabels(dec_qy)
# plt.savefig("dec.png")
# plt.show()


# tryout
# x11slct = rsp.select(k=rsp.indices(fstind), q=x1_ind, pred=rsp.equals, causal=False)
# x21slct = rsp.select(k=rsp.indices(fstind), q=x2_ind, pred=rsp.equals, causal=False)
# qy1slct = rsp.select(k=rsp.indices(fstind), q=qy_ind, pred=rsp.equals, causal=False)

# x12slct = rsp.select(k=rsp.indices(fstind), q=x1_ind+1, pred=rsp.equals, causal=False)
# x22slct = rsp.select(k=rsp.indices(fstind), q=x2_ind+1, pred=rsp.equals, causal=False)
# qy2slct = rsp.select(k=rsp.indices(fstind), q=qy_ind+1, pred=rsp.equals, causal=False)

# x13slct = rsp.select(k=rsp.indices(fstind), q=x1_ind+2, pred=rsp.equals, causal=False)
# x23slct = rsp.select(k=rsp.indices(fstind), q=x2_ind+2, pred=rsp.equals, causal=False)
# qy3slct = rsp.select(k=rsp.indices(fstind), q=qy_ind+2, pred=rsp.equals, causal=False)

# id = rsp.select(k=rsp.indices(s), q=rsp.indices(s), pred=rsp.equals, causal=False)

# x1allslct = (x11slct | x12slct | x13slct) & id
# x2allslct = (x21slct | x22slct | x23slct) & id
# qyallslct = (qy1slct | qy2slct | qy3slct) & id

# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(x1allslct)
# axs[1].imshow(x2allslct)
# axs[2].imshow(qyallslct)
# for ax in axs:
#     ax.set_xticks(range(len(s)))
#     ax.set_xticklabels(s)
#     ax.set_yticks(range(len(s)))
#     ax.set_yticklabels(s)
# plt.savefig("layer2.png")
# plt.show()