from tools.visit import Visit
tool = Visit()
out, _ = tool.call({
  "url": ["https://en.wikipedia.org/wiki/Sportforum_Hohensch%C3%B6nhausen"],
  "goal": "Find the spectator count in the 16 August 1986 opening match"
})
print(out[:2000])