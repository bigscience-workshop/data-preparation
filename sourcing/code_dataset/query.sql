SELECT
  f.repo_name,
  f.path,
  c.copies,
  c.size,
  c.content,
  l.license
FROM (
  SELECT
    f.*,
    ROW_NUMBER() OVER (PARTITION BY id) AS seqnum
  FROM
    `bigquery-public-data.github_repos.files` AS f) f
JOIN
  `bigquery-public-data.github_repos.contents` AS c
ON
  f.id = c.id
  AND seqnum=1
JOIN
  `bigquery-public-data.github_repos.licenses` AS l
ON
  f.repo_name = l.repo_name
WHERE
  NOT c.binary
  AND ((f.path LIKE '%.cs'
      OR f.path LIKE '%.cpp'
      OR f.path LIKE '%.hpp'
      OR f.path LIKE '%.c++'
      OR f.path LIKE '%.h++'
      OR f.path LIKE '%.cc'
      OR f.path LIKE '%.hh'
      OR f.path LIKE '%.C'
      OR f.path LIKE '%.H'
      OR f.path LIKE '%.go'
      OR f.path LIKE '%.java'
      OR f.path LIKE '%.js'
      OR f.path LIKE '%.lua'
      OR f.path LIKE '%.php'
      OR f.path LIKE '%.php3'
      OR f.path LIKE '%.php4'
      OR f.path LIKE '%.php5'
      OR f.path LIKE '%.phps'
      OR f.path LIKE '%.phpt'
      OR f.path LIKE '%.py'
      OR f.path LIKE '%.rb'
      OR f.path LIKE '%.rs'
      OR f.path LIKE '%.scala'
      OR f.path LIKE '%.ts'
      OR f.path LIKE '%.tsx')
    AND (c.size BETWEEN 0
      AND 1048575))