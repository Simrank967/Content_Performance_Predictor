SELECT
    channel_title,
    COUNT(*) AS total_trending,
    ROUND(AVG(views),2) AS avg_views,
    ROUND(AVG(likes),2) AS avg_likes
FROM youtube_videos
GROUP BY channel_title
HAVING total_trending >= 5
ORDER BY avg_views DESC
LIMIT 20;