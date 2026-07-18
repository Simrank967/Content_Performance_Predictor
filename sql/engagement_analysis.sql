SELECT
    title,
    channel_title,
    views,
    likes,
    comment_count,
    ROUND(((likes + comment_count) * 100.0) / views,2) AS engagement_rate
FROM youtube_videos
WHERE views > 0
ORDER BY engagement_rate DESC
LIMIT 20;