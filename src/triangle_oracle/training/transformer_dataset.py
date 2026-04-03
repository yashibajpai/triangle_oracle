import networkx as nx  
import pandas as pd  


SPECIAL_TOKENS = {  # define special tokens for sequence structure
    "[PAD]": 0,  # padding token
    "[EDGE]": 1,  # marks start of edge
    "[U]": 2,  # marks node u
    "[V]": 3,  # marks node v
    "[NU]": 4,  # neighbors of u
    "[NV]": 5,  # neighbors of v
    "[SEP]": 6,  # separator token
}


def build_node_vocab(g: nx.Graph) -> dict:  # create mapping from nodes to token ids
    vocab = dict(SPECIAL_TOKENS)  # start with special tokens
    next_id = max(vocab.values()) + 1  # next available id

    for node in g.nodes():  # iterate through nodes
        if node not in vocab:  # ensure not already assigned
            vocab[node] = next_id  # assign id
            next_id += 1  # increment id

    return vocab  # return mapping


def serialize_edge_neighborhood(  # convert edge neighborhood into token sequence
    g: nx.Graph,
    u,
    v,
    vocab: dict,
    max_neighbors: int = 20,
) -> list[int]:

    neighbors_u = sorted(list(g.neighbors(u)))[:max_neighbors]  # get neighbors of u
    neighbors_v = sorted(list(g.neighbors(v)))[:max_neighbors]  # get neighbors of v

    seq = [  # initialize sequence
        vocab["[EDGE]"],  # edge marker
        vocab["[U]"], vocab[u],  # node u
        vocab["[V]"], vocab[v],  # node v
        vocab["[SEP]"],  # separator
        vocab["[NU]"],  # start neighbors of u
    ]

    seq.extend(vocab[n] for n in neighbors_u if n in vocab)  # add neighbors of u

    seq.append(vocab["[SEP]"])  # separator between groups

    seq.append(vocab["[NV]"])  # start neighbors of v
    seq.extend(vocab[n] for n in neighbors_v if n in vocab)  # add neighbors of v

    return seq  # return token list


def build_transformer_dataframe(  # build dataframe for transformer training
    g: nx.Graph,
    edge_df: pd.DataFrame,
    vocab: dict,
    max_neighbors: int = 20,
) -> pd.DataFrame:

    rows = []  # store rows

    for _, row in edge_df.iterrows():  # iterate edges
        u = row["u"]  # source node
        v = row["v"]  # target node
        y = row["edge_heaviness"]  # label

        token_ids = serialize_edge_neighborhood(  # build token sequence
            g=g,
            u=u,
            v=v,
            vocab=vocab,
            max_neighbors=max_neighbors,
        )

        rows.append({  # append row
            "u": u,
            "v": v,
            "token_ids": token_ids,
            "edge_heaviness": y,
        })

    return pd.DataFrame(rows)  # return dataframe