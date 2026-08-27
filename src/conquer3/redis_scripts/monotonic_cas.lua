-- KEYS[1]=key ARGV[1]=json ARGV[2]=new_updated_at_us ARGV[3]=ttl_s
local cur = redis.call('GET', KEYS[1])
if cur then
  local ok, prev = pcall(cjson.decode, cur)
  if ok and tonumber(prev.updated_at_us) > tonumber(ARGV[2]) then return 0 end
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[3]))
return 1
