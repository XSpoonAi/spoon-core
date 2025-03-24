import os
from logging import getLogger
from typing import Dict, List

import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1Session

logger = getLogger(__name__)

class TwitterClient:
    def __init__(self):
        pass
    
    def _make_request(self, method: str, endpoint: str,use_bearer: bool = False, stream: bool = False, **kwargs) -> dict:
        """
        Make a request to the Twitter API with error handling

        Args:
            method: HTTP method ('get', 'post', etc.)
            endpoint: API endpoint path
            **kwargs: Additional request parameters

        Returns:
            Dict containing the API response (or raw response if stream=True)
        """
        logger.debug(f"Making {method.upper()} request to {endpoint}")
        try:
            full_url = f"https://api.twitter.com/2/{endpoint.lstrip('/')}"

            if use_bearer:
                response = requests.request(
                    method=method.lower(),
                    url=full_url,
                    auth=self._bearer_oauth,
                    stream=stream,
                    **kwargs
                )
            else:
                oauth = self._get_oauth()
                response = getattr(oauth, method.lower())(full_url, **kwargs)

            if not stream and response.status_code not in [200, 201]:
                logger.error(
                    f"Request failed: {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"Request failed with status {response.status_code}: {response.text}"
                )

            logger.debug(f"Request successful: {response.status_code}")

            if stream:
                return response
        
            return response.json()

        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")
    
    
    def _get_oauth(self) -> OAuth1Session:
        """Get or create OAuth session using stored credentials"""
        if self._oauth_session is None:
            logger.debug("Creating new OAuth session")
            try:
                credentials = self._get_credentials()
                self._oauth_session = OAuth1Session(
                    credentials['TWITTER_CONSUMER_KEY'],
                    client_secret=credentials['TWITTER_CONSUMER_SECRET'],
                    resource_owner_key=credentials['TWITTER_ACCESS_TOKEN'],
                    resource_owner_secret=credentials[
                        'TWITTER_ACCESS_TOKEN_SECRET'],
                )
                logger.debug("OAuth session created successfully")
            except Exception as e:
                logger.error(f"Failed to create OAuth session: {str(e)}")
                raise

        return self._oauth_session

    def _get_credentials(self) -> Dict[str, str]:
        """Get Twitter credentials from environment with validation"""
        logger.debug("Retrieving Twitter credentials")
        load_dotenv()

        required_vars = {
            'TWITTER_CONSUMER_KEY': os.getenv('TWITTER_CONSUMER_KEY'),
            'TWITTER_CONSUMER_SECRET': os.getenv('TWITTER_CONSUMER_SECRET'),
            'TWITTER_ACCESS_TOKEN': os.getenv('TWITTER_ACCESS_TOKEN'),
            'TWITTER_ACCESS_TOKEN_SECRET': os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
            'TWITTER_USER_ID': os.getenv('TWITTER_USER_ID')
        }

        optional_vars = {'TWITTER_BEARER_TOKEN'} # Bearer Token is used for streaming, Twitter premium plan is required

        credentials = {}
        missing = []

        for env_var, description in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                missing.append(description)
            credentials[env_var] = value

        if missing:
            error_msg = f"Missing Twitter credentials: {', '.join(missing)}"
            raise Exception(error_msg)
        
        for env_var in optional_vars:
            credentials[env_var] = os.getenv(env_var)

        logger.debug("All required credentials found")
        return credentials
    
    def read_timeline(self, count: int = None, **kwargs) -> list:
        """Read tweets from the user's timeline"""
        if count is None:
            count = self.config["timeline_read_count"]
            
        logger.debug(f"Reading timeline, count: {count}")
        credentials = self._get_credentials()

        params = {
            "tweet.fields": "created_at,author_id,attachments",
            "expansions": "author_id",
            "user.fields": "name,username",
            "max_results": count
        }

        response = self._make_request(
            'get',
            f"users/{credentials['TWITTER_USER_ID']}/timelines/reverse_chronological",
            params=params
        )

        tweets = response.get("data", [])
        user_info = response.get("includes", {}).get("users", [])

        user_dict = {
            user['id']: {
                'name': user['name'],
                'username': user['username']
            }
            for user in user_info
        }

        for tweet in tweets:
            author_id = tweet['author_id']
            author_info = user_dict.get(author_id, {
                'name': "Unknown",
                'username': "Unknown"
            })
            tweet.update({
                'author_name': author_info['name'],
                'author_username': author_info['username']
            })

        logger.debug(f"Retrieved {len(tweets)} tweets")
        return tweets
    
    def post_tweet(self, message: str, **kwargs) -> dict:
        """Post a new tweet"""
        logger.debug("Posting new tweet")
        self._validate_tweet_text(message)

        response = self._make_request('post', 'tweets', json={'text': message})

        logger.info("Tweet posted successfully")
        return response
    def reply_to_tweet(self, tweet_id: str, message: str, **kwargs) -> dict:
        """Reply to an existing tweet"""
        logger.debug(f"Replying to tweet {tweet_id}")
        self._validate_tweet_text(message, "Reply")

        response = self._make_request('post',
                                      'tweets',
                                      json={
                                          'text': message,
                                          'reply': {
                                              'in_reply_to_tweet_id': tweet_id
                                          }
                                      })

        logger.info("Reply posted successfully")
        return response

    def like_tweet(self, tweet_id: str, **kwargs) -> dict:
        """Like a tweet"""
        logger.debug(f"Liking tweet {tweet_id}")
        credentials = self._get_credentials()

        response = self._make_request(
            'post',
            f"users/{credentials['TWITTER_USER_ID']}/likes",
            json={'tweet_id': tweet_id})

        logger.info("Tweet liked successfully")
        return response
    
    def get_tweet_replies(self, tweet_id: str, count: int = 10, **kwargs) -> List[dict]:
        """Fetch replies to a specific tweet"""
        logger.debug(f"Fetching replies for tweet {tweet_id}, count: {count}")
        
        params = {
            "query": f"conversation_id:{tweet_id} is:reply",
            "tweet.fields": "author_id,created_at,text",
            "max_results": min(count, 100)
        }
        
        response = self._make_request('get', 'tweets/search/recent', params=params)
        replies = response.get("data", [])
        
        logger.info(f"Retrieved {len(replies)} replies")
        return replies

    def send(self, message: str, tags: Optional[List[str]] = None, **kwargs) -> bool:
        """
        发送Twitter通知消息
        
        这个方法专门用于监控系统的通知功能，它将消息作为推文发布
        
        Args:
            message: 通知消息内容
            tags: 要附加的标签列表
            **kwargs: 其他参数
        
        Returns:
            bool: 是否发送成功
        """
        try:
            # 检查消息长度并截断
            max_length = 280
            if len(message) > max_length:
                message = message[:max_length-3] + "..."
            
            # 添加标签
            if tags is None:
                tags = kwargs.get("tags", ["#CryptoAlert", "#TradingAlert"])
            
            # 如果消息中没有标签且有空间，添加它们
            if all(tag not in message for tag in tags) and len(message) + sum(len(tag) + 1 for tag in tags) <= max_length:
                message += " " + " ".join(tags)
            
            # 发送推文
            self.post_tweet(message)
            logger.info("Twitter notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Twitter notification: {str(e)}")
            return False