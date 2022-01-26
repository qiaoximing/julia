module Dataset

using Random
export test_dataset

dict = [
"a",
"about",
"all",
"also",
"and",
"as",
"at",
"be",
"because",
"but",
"by",
"can",
"come",
"could",
"day",
"do",
"even",
"find",
"first",
"for",
"from",
"get",
"give",
"go",
"have",
"he",
"her",
"here",
"him",
"his",
"how",
"I",
"if",
"in",
"into",
"it",
"its",
"just",
"know",
"like",
"look",
"make",
"man",
"many",
"me",
"more",
"my",
"new",
"no",
"not",
"now",
"of",
"on",
"one",
"only",
"or",
"other",
"our",
"out",
"people",
"say",
"see",
"she",
"so",
"some",
"take",
"tell",
"than",
"that",
"the",
"their",
"them",
"then",
"there",
"these",
"they",
"thing",
"think",
"this",
"those",
"time",
"to",
"two",
"up",
"use",
"very",
"want",
"way",
"we",
"well",
"what",
"when",
"which",
"who",
"will",
"with",
"would",
"year",
"you",
"your"][1:3]
dict=["big", "you","hel"]

function sentence()
    s = ""
    # while true
    #     s *= rand(dict) * "_"
    #     if rand() < 0.5
    #         break
    #     end
    # end
    s *= rand(dict)
    return s
end

function test_dataset(n::Int, m::Int)
    ds = []
    for i in 1:n
        s = sentence()
        for j in 1:m
            push!(ds, s)
        end
    end
    return ds
end

end