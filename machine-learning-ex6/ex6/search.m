function idx = search(dict, word)

m = ismember(dict, word);

if any(m)
  idx = find(m);
else
  idx = 0;
end



end