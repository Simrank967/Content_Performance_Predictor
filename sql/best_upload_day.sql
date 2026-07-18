SELECT
    publish_day,
    COUNT(*) AS total_videos,
    ROUND(AVG(views),2) AS avg_views,
    ROUND(AVG(likes),2) AS avg_likes,
    ROUND(AVG(comment_count),2) AS avg_comments
FROM youtube_videos
GROUP BY publish_day
ORDER BY avg_views DESC;