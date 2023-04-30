from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional
import json

from tree_proccesor import SearchRule, SubtreeParaphraser, Paraphraser

class Params(BaseModel):
    tree: str
    limit: Optional[int] = Query(default=20)

app = FastAPI()

@app.get("/paraphrase")
def paraphrase(params: Params = Depends()):

    if not params.tree:
        raise HTTPException(status_code=400, detail="Bad Request: Tree string is required")
    
    search_rule = SearchRule()
    subtree_paraphraser = SubtreeParaphraser()
    paraphraser = Paraphraser(params.tree, search_rule, subtree_paraphraser)

    paraphrase_trees = [{"tree": tree} for tree in paraphraser.get_all_trees()][:params.limit]
    trees_dict = {"paraphrases": paraphrase_trees}

    with open('trees_responce.json', 'w', encoding="utf-8") as f:
        json.dump(trees_dict, f, ensure_ascii=False)

    return trees_dict