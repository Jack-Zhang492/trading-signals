from core.data_fetcher import get_main_board_stocks

s = get_main_board_stocks(None)
print(f"Total: {len(s)}")
print(s.head(5))

codes = s["code"].tolist()
kcb = sum(1 for c in codes if c.startswith("688") or c.startswith("689"))
bj = sum(1 for c in codes if c.startswith("8") or c.startswith("4"))
st = sum(1 for n in s["name"] if "ST" in str(n))
print(f"科创板(688/689): {kcb}")
print(f"北交所/新三板(8/4): {bj}")
print(f"ST股: {st}")
print(f"有效股票: {len(s) - kcb - bj - st}")
