"""
Multi-provider API client with automatic fallback
Tries: OpenAI → Groq → Anthropic
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class MultiProviderClient:
    """
    Universal LLM client that tries multiple providers with fallback
    """
    
    def __init__(self):
        self.providers = []
        self.current_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers based on API keys in .env"""
        
        # Try OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                # Test with a small request to verify quota
                try:
                    test = client.chat.completions.create(
                        model='gpt-4o-mini',
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=1
                    )
                    self.providers.append({
                        'name': 'OpenAI',
                        'client': client,
                        'model': 'gpt-4o-mini',
                        'type': 'openai'
                    })
                    print("[OK] OpenAI initialized")
                except Exception as quota_err:
                    if '429' in str(quota_err) or 'quota' in str(quota_err).lower():
                        print("[WARN] OpenAI quota exceeded, skipping")
                    else:
                        raise
            except Exception as e:
                print(f"[WARN] OpenAI initialization failed: {e}")
        
        # Try Groq
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                self.providers.append({
                    'name': 'Groq',
                    'client': client,
                    'model': 'llama-3.3-70b-versatile',
                    'type': 'groq'
                })
                print("[OK] Groq initialized")
            except Exception as e:
                print(f"[WARN] Groq initialization failed: {e}")
        
        # Try Anthropic
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                self.providers.append({
                    'name': 'Anthropic',
                    'client': client,
                    'model': 'claude-3-5-haiku-20241022',
                    'type': 'anthropic'
                })
                print("[OK] Anthropic initialized")
            except Exception as e:
                print(f"[WARN] Anthropic initialization failed: {e}")
        
        if not self.providers:
            raise Exception("No API providers available! Add at least one API key to .env")
        
        self.current_provider = self.providers[0]
        print(f"\n[INFO] Using {self.current_provider['name']} as primary provider")
        if len(self.providers) > 1:
            fallbacks = [p['name'] for p in self.providers[1:]]
            print(f"   Fallbacks: {', '.join(fallbacks)}")
    
    def chat_completion(self, messages: List[Dict], temperature: float = 0.7, 
                       max_tokens: int = 500) -> str:
        """
        Create chat completion with automatic fallback
        """
        last_error = None
        
        for provider in self.providers:
            try:
                if provider['type'] == 'openai' or provider['type'] == 'groq':
                    response = provider['client'].chat.completions.create(
                        model=provider['model'],
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content.strip()
                
                elif provider['type'] == 'anthropic':
                    # Anthropic has different message format
                    system_msg = None
                    conv_messages = []
                    
                    for msg in messages:
                        if msg['role'] == 'system':
                            system_msg = msg['content']
                        else:
                            conv_messages.append({
                                'role': msg['role'],
                                'content': msg['content']
                            })
                    
                    response = provider['client'].messages.create(
                        model=provider['model'],
                        system=system_msg,
                        messages=conv_messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.content[0].text.strip()
            
            except Exception as e:
                last_error = e
                # Only print if it's the last provider
                if provider == self.providers[-1]:
                    print(f"[WARN] All providers failed. Last error: {str(e)[:100]}")
                # Silently try next provider
                continue
        
        # All providers failed
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def get_provider_name(self) -> str:
        """Get current provider name"""
        return self.current_provider['name'] if self.current_provider else "None"


class EmbeddingClient:
    """
    Embedding client with fallback
    """
    
    def __init__(self):
        self.provider = None
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize embedding provider"""
        
        # Try OpenAI first
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                from openai import OpenAI
                self.provider = OpenAI(api_key=openai_key)
                self.model = 'text-embedding-3-small'
                self.type = 'openai'
                self.dimension = 1536
                print("[OK] Using OpenAI embeddings")
                return
            except:
                pass
        
        # Fallback to local sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading local embedding model...")
            self.provider = SentenceTransformer('all-MiniLM-L6-v2')
            self.type = 'local'
            self.dimension = 384
            print("[OK] Using local embeddings (sentence-transformers)")
        except Exception as e:
            raise Exception(f"Failed to initialize embeddings: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for single text"""
        try:
            if self.type == 'openai':
                response = self.provider.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
            else:  # local
                return self.provider.encode(text).tolist()
        except Exception as e:
            # If OpenAI fails, fall back to local
            if self.type == 'openai':
                print(f"[WARN] OpenAI embeddings failed, switching to local model...")
                from sentence_transformers import SentenceTransformer
                self.provider = SentenceTransformer('all-MiniLM-L6-v2')
                self.type = 'local'
                self.dimension = 384
                return self.provider.encode(text).tolist()
            print(f"Error getting embedding: {e}")
            return [0] * self.dimension
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        if self.type == 'openai':
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    response = self.provider.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    print(f"  Embedded {len(all_embeddings)}/{len(texts)} texts...")
                except Exception as e:
                    print(f"[WARN] OpenAI embeddings failed: {str(e)[:100]}")
                    print(f"  Switching to local model for remaining {len(texts) - len(all_embeddings)} texts...")
                    
                    # Switch to local for the rest
                    from sentence_transformers import SentenceTransformer
                    self.provider = SentenceTransformer('all-MiniLM-L6-v2')
                    self.type = 'local'
                    self.dimension = 384
                    
                    # Process remaining texts locally
                    remaining_texts = texts[len(all_embeddings):]
                    for j in range(0, len(remaining_texts), batch_size):
                        batch = remaining_texts[j:j + batch_size]
                        embeddings = self.provider.encode(batch).tolist()
                        all_embeddings.extend(embeddings)
                        print(f"  Embedded {len(all_embeddings)}/{len(texts)} texts (local)...")
                    break
            return all_embeddings
        else:  # local
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self.provider.encode(batch).tolist()
                all_embeddings.extend(embeddings)
                print(f"  Embedded {len(all_embeddings)}/{len(texts)} texts...")
            return all_embeddings


# Global instances
try:
    llm_client = MultiProviderClient()
    embedding_client = EmbeddingClient()
except Exception as e:
    print(f"[ERROR] Error initializing API clients: {e}")
    print("\nMake sure you have at least one API key in your .env file:")
    print("  OPENAI_API_KEY=sk-...")
    print("  GROQ_API_KEY=gsk_...")
    print("  ANTHROPIC_API_KEY=sk-ant-...")
    raise


if __name__ == "__main__":
    # Test the client
    print("\n" + "="*60)
    print("TESTING MULTI-PROVIDER CLIENT")
    print("="*60)
    
    print(f"\n[INFO] Active LLM Provider: {llm_client.get_provider_name()}")
    print(f"[INFO] Embedding Type: {embedding_client.type}")
    print(f"   Dimension: {embedding_client.dimension}")
    
    # Test chat
    print("\n" + "="*60)
    print("Testing chat completion...")
    try:
        response = llm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello!' in one word."}
            ],
            max_tokens=10
        )
        print(f"Response: {response}")
        print("[OK] Chat completion working!")
    except Exception as e:
        print(f"[FAIL] Chat failed: {e}")
    
    # Test embedding
    print("\n" + "="*60)
    print("Testing embeddings...")
    try:
        embedding = embedding_client.get_embedding("test text")
        print(f"Embedding dimension: {len(embedding)}")
        print("[OK] Embeddings working!")
    except Exception as e:
        print(f"[FAIL] Embedding failed: {e}")