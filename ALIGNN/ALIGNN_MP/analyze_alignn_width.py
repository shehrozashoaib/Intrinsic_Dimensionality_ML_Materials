"""Analyze ALIGNN param scaling when widening the CONVOLUTION layers.

ALIGNN's conv stack (from alignn_atomwise.py):
  alignn_layers = ALIGNNConv(hidden_features, hidden_features)
                   -> node_update = EdgeGatedGraphConv(h, h)
                   -> edge_update = EdgeGatedGraphConv(h, h)
  gcn_layers    = EdgeGatedGraphConv(hidden_features, hidden_features)
  EdgeGatedGraphConv = 5x Linear(h,h) (src_gate,dst_gate,edge_gate,src_update,dst_update) + 2x LayerNorm

CRITICAL: the conv width IS `hidden_features`, which is ALSO the output width of the atom/edge/angle
embeddings and the input width of the readout fc. Unlike DimeNet++ (which has a genuine internal
bottleneck `int_emb_size` inside the interaction block, separate from `emb_size`), ALIGNN has NO
conv-only width knob. So raising hidden_features widens conv AND embeddings AND readout.

This script quantifies exactly how much of the growth is conv vs non-conv.
"""
import torch
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig

BASE = dict(name="alignn_atomwise", alignn_layers=1, gcn_layers=1,
            atom_input_features=92, edge_input_features=80, triplet_input_features=40,
            embedding_features=64, output_features=1, classification=False,
            calculate_gradient=False, atomwise_output_features=0, energy_mult_natoms=False)


def group(n):
    n = n.lower()
    if "alignn_layers" in n or "gcn_layers" in n:
        return "CONV (alignn+gcn layers)"
    if "embedding" in n:
        return "embeddings (atom/edge/angle)"
    if n.startswith("fc") or "readout" in n:
        return "readout / fc"
    return "other"


rows = {}
for mult, h in [("1x", 64), ("2x", 128), ("4x", 256)]:
    m = ALIGNNAtomWise(ALIGNNAtomWiseConfig(hidden_features=h, **BASE))
    per, total = {}, 0
    for name, p in m.named_parameters():
        c = p.numel(); total += c
        per[group(name)] = per.get(group(name), 0) + c
    rows[mult] = (h, total, per)
    print(f"\n=== hidden_features={h} ({mult}) ===")
    print(f"  TOTAL D = {total:,}")
    for k, v in sorted(per.items(), key=lambda x: -x[1]):
        print(f"    {k:30s} {v:>9,}  ({100*v/total:.1f}%)")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
b = rows["1x"][1]
for mult in ("1x", "2x", "4x"):
    h, tot, _ = rows[mult]
    print(f"  hidden_features={h:4d} ({mult}): D={tot:>8,}   {tot/b:.2f}x vs 1x")
print("\nGrowth breakdown (what actually gets wider):")
for k in rows["1x"][2]:
    v1, v2, v4 = (rows[m][2].get(k, 0) for m in ("1x", "2x", "4x"))
    tag = "GROWS" if v4 != v1 else "fixed"
    print(f"  {k:30s} 1x={v1:>8,}  2x={v2:>8,}  4x={v4:>8,}   {tag}")
c1, c4 = rows["1x"][2].get("CONV (alignn+gcn layers)", 0), rows["4x"][2].get("CONV (alignn+gcn layers)", 0)
g_tot = rows["4x"][1] - rows["1x"][1]
g_conv = c4 - c1
print(f"\n  Of the +{g_tot:,} params added going 1x->4x, {g_conv:,} ({100*g_conv/g_tot:.1f}%) are CONV.")
print(f"  => conv-dominated? {'YES' if g_conv/g_tot > 0.9 else 'NO — embeddings/readout also grow'}")
