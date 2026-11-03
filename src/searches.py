"""Every search query the benchmark runs. The parameters are $1 vector, $2 limit, $3 question."""

USER_FILTER = "owner = auth.uid()"


def _where(*conditions):
    conditions = [c for c in conditions if c]
    return "WHERE " + " AND ".join(conditions) if conditions else ""


def _tsquery(operator):
    if operator == "and":
        return "plainto_tsquery('english', $3)"
    if operator == "or":
        return "replace(plainto_tsquery('english', $3)::text, '&', '|')::tsquery"
    raise ValueError(f"unknown operator {operator}")


def vector(table, user_filter=""):
    return f"SELECT id FROM {table} {_where(user_filter)} ORDER BY embedding <=> $1 LIMIT $2"


def keyword(table, operator="and", user_filter="", min_score=0.0):
    tsq = _tsquery(operator)
    return f"""SELECT id FROM (
        SELECT id, ts_rank_cd(content_tsv, {tsq}) AS score FROM {table}
        {_where(f"content_tsv @@ {tsq}", user_filter)}
    ) scored WHERE score >= {float(min_score)} ORDER BY score DESC, id LIMIT $2"""


def hybrid(table, operator="and", user_filter="", min_score=0.0, candidates=50, rrf_k=60):
    """Reciprocal Rank Fusion over the vector and keyword rankings. Only ranks are combined,
    because a cosine distance and a text rank aren't comparable."""
    tsq = _tsquery(operator)
    candidates, rrf_k = int(candidates), int(rrf_k)
    return f"""WITH vec AS (
        SELECT id, row_number() OVER () AS rnk FROM (
            SELECT id FROM {table} {_where(user_filter)} ORDER BY embedding <=> $1 LIMIT {candidates}
        ) nearest
    ), lex AS (
        SELECT id, row_number() OVER (ORDER BY score DESC, id) AS rnk FROM (
            SELECT id, ts_rank_cd(content_tsv, {tsq}) AS score FROM {table}
            {_where(f"content_tsv @@ {tsq}", user_filter)}
            ORDER BY score DESC, id LIMIT {candidates}
        ) scored WHERE score >= {float(min_score)}
    )
    SELECT id FROM vec FULL JOIN lex USING (id)
    ORDER BY coalesce(1.0 / ({rrf_k} + vec.rnk), 0) + coalesce(1.0 / ({rrf_k} + lex.rnk), 0) DESC, id
    LIMIT $2"""
