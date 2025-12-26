# Explanation

## Rant

AI is well known to have "Jagged Capability frontier". It can be very capable in some area but still have weaknesses that is unusual, or quirky, from a human perspective. Perhaps it is best to remember that although LLM's human-AI interaction can feel human at time, there is at least some aspects of AI that is more like an alien intelligence - it think, but differently from a human.

## Background

While developing the Age of empire minigame clone (refer to `GLM_47_tests` folder in this same repo), the code is getting long, so I extracted the web UI part, and try various AI. The results are surprisingly under-performing compared to expectations, especially given that:

- Vanilla web UI/frontend dev should be the easiest type of task compared to some of the more niche one such as low level hardware programming (eg custom driver), or complex CS algorithm engineering challenge (that requires integrating complex algorithm in the context of some larger system)
- The same AI have been performing very well in the more challenging task before, then it suddenly underperform in the "easier" task (hence the remarks above that AI may have vastly different "performance curve" / "workload characteristics" compared to human)

## Attempts/Records

Attempt 1: Deepseek R1 (I'm not sure about the true model identity or scaffolding of this one, been using online service)

- Only gave high level code, not instruction following well
- I do like its progress bar though

Attempt 2: GLM 4.7 (The original)

- Maybe I underspecified the task - it changed too much things
- But otherwise it did well - especially the tech tree and the top civ banner

Attempt 3: Minimax M2.1

- After the last two failure, we try to give more specific feedback and reflection on previous failures
- Now it finally does well

## Analysis

Possibly, its due to a combination of:

- Long context general performance degradation
- LLM-get-lost effect (misdirection, context poisoning, etc) (especially considering we attached transcript of previous failures)
- Under-specification and ambiguity
- Also LLM API provider stability issues

