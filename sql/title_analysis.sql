SELECT
    LENGTH(title) AS title_length,
    COUNT(*) AS total_videos,
    ROUND(AVG(views),2) AS avg_views,
    ROUND(AVG(likes),2) AS avg_likes
FROM youtube_videos
GROUP BY title_length
ORDER BY title_length;