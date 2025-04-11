import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [socket, setSocket] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState("Connecting...");
  const [connectionAttempts, setConnectionAttempts] = useState(0);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const MAX_RECONNECT_ATTEMPTS = 5;
  const reconnectTimeoutRef = useRef(null);
  const typingTimeoutRef = useRef(null);
  const processingTimeoutRef = useRef(null);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Auto-resize textarea as user types
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${inputRef.current.scrollHeight}px`;
    }
  }, [input]);

  // Setup WebSocket connection with robust reconnection logic
  useEffect(() => {
    console.log("Setting up WebSocket connection");
    connectWebSocket();
    
    // Cleanup function
    return () => {
      if (socket) {
        try {
          // Only close if the socket is actually open
          if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
            console.log("Cleaning up WebSocket connection");
            socket.close(1000, "Client disconnecting normally");
          }
        } catch (err) {
          console.error("Error closing WebSocket:", err);
        }
      }
      // Clear all timeouts
      [reconnectTimeoutRef, typingTimeoutRef, processingTimeoutRef].forEach(ref => {
        if (ref.current) {
          clearTimeout(ref.current);
        }
      });
    };
  }, [connectionAttempts]);

  const connectWebSocket = () => {
    try {
      // Close any existing socket
      if (socket) {
        try {
          if (socket.readyState !== WebSocket.CLOSED && socket.readyState !== WebSocket.CLOSING) {
            socket.close(1000, "Reconnecting");
          }
        } catch (err) {
          console.error("Error closing existing WebSocket:", err);
        }
      }
      
      // Get WebSocket URL from environment if available, otherwise use default
      const wsUrl = process.env.REACT_APP_BACKEND_URL || "ws://localhost:8000/ws";
      console.log(`Connecting to WebSocket at ${wsUrl}`);
      const ws = new WebSocket(wsUrl);
      
      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          console.log("WebSocket connection timeout");
          ws.close();
          setConnectionStatus("Connection Timeout");
          
          // Trigger reconnect
          if (connectionAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectTimeoutRef.current = setTimeout(() => {
              setConnectionAttempts(prev => prev + 1);
            }, 2000);
          }
        }
      }, 5000);
      
      ws.onopen = () => {
        console.log("WebSocket connection established");
        clearTimeout(connectionTimeout);
        setConnectionStatus("Connected");
        setMessages(prev => [...prev, { 
          id: Date.now(), 
          sender: "system", 
          text: "Connected to server successfully." 
        }]);
        setConnectionAttempts(0); // Reset connection attempts on success
      };
      
      ws.onmessage = (event) => {
        console.log("Message received:", event.data);
        try {
          // Try to parse as JSON first
          const jsonData = JSON.parse(event.data);
          
          switch (jsonData.type) {
            case "error":
              // Handle error message
              setIsTyping(false);
              if (typingTimeoutRef.current) {
                clearTimeout(typingTimeoutRef.current);
              }
              if (processingTimeoutRef.current) {
                clearTimeout(processingTimeoutRef.current);
              }
              
              setMessages(prev => [...prev, { 
                id: Date.now(), 
                sender: "system", 
                text: jsonData.message,
                isError: true 
              }]);
              break;
              
            case "response":
              // Handle normal response
              setIsTyping(false);
              if (typingTimeoutRef.current) {
                clearTimeout(typingTimeoutRef.current);
              }
              if (processingTimeoutRef.current) {
                clearTimeout(processingTimeoutRef.current);
              }
              
              setMessages(prev => [...prev, { 
                id: Date.now(), 
                sender: "bot", 
                text: jsonData.message 
              }]);
              break;
              
            case "status":
              // Handle status updates (AI is thinking, processing tools, etc.)
              // Don't clear typing indicator, just show the status
              setMessages(prev => [...prev, { 
                id: Date.now(), 
                sender: "system", 
                text: jsonData.message,
                isStatus: true
              }]);
              break;
              
            case "warning":
              // Handle warning messages
              setMessages(prev => [...prev, { 
                id: Date.now(), 
                sender: "system", 
                text: jsonData.message,
                isWarning: true 
              }]);
              break;
              
            default:
              // Handle any other JSON format
              setIsTyping(false);
              setMessages(prev => [...prev, { 
                id: Date.now(), 
                sender: "bot", 
                text: jsonData.message || JSON.stringify(jsonData)
              }]);
          }
        } catch (e) {
          // Not JSON, treat as plain text
          console.log("Received non-JSON message:", e);
          setIsTyping(false);
          setMessages(prev => [...prev, { 
            id: Date.now(), 
            sender: "bot", 
            text: event.data 
          }]);
        }
      };
      
      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setConnectionStatus("Connection Error");
      };
      
      ws.onclose = (event) => {
        console.log("WebSocket connection closed", event);
        setConnectionStatus("Disconnected");
        
        // Stop typing indicator if it's active
        setIsTyping(false);
        
        // Only attempt to reconnect if we haven't exceeded maximum attempts
        if (connectionAttempts < MAX_RECONNECT_ATTEMPTS) {
          setMessages(prev => [...prev, { 
            id: Date.now(), 
            sender: "system", 
            text: `Connection lost. Attempting to reconnect (${connectionAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})...` 
          }]);
          
          // Exponential backoff for reconnection attempts
          const timeout = Math.min(1000 * (2 ** connectionAttempts), 10000);
          reconnectTimeoutRef.current = setTimeout(() => {
            setConnectionAttempts(prev => prev + 1);
          }, timeout);
        } else {
          setMessages(prev => [...prev, { 
            id: Date.now(), 
            sender: "system", 
            text: "Maximum reconnection attempts reached. Please refresh the page." 
          }]);
        }
      };
      
      setSocket(ws);
    } catch (err) {
      console.error("Error setting up WebSocket:", err);
      setConnectionStatus("Connection Failed");
    }
  };

  const sendMessage = () => {
    if (socket && socket.readyState === WebSocket.OPEN && input.trim()) {
      const messageToSend = input.trim();
      
      // Add user message to chat
      setMessages(prev => [...prev, { 
        id: Date.now(), 
        sender: "user", 
        text: messageToSend 
      }]);
      
      // Show typing indicator
      setIsTyping(true);
      
      // Send the message to the server
      socket.send(messageToSend);
      
      // Clear input field
      setInput("");
      
      // Clear any existing timeouts
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      if (processingTimeoutRef.current) {
        clearTimeout(processingTimeoutRef.current);
      }
      
      // Set first timeout for initial processing message (20 seconds)
      typingTimeoutRef.current = setTimeout(() => {
        setMessages(prev => [...prev, { 
          id: Date.now(), 
          sender: "system", 
          text: "Still processing your request...",
          isStatus: true
        }]);
        
        // Set a longer timeout for warning message (60 seconds)
        processingTimeoutRef.current = setTimeout(() => {
          setIsTyping(false); // Remove typing indicator after 60 seconds
          setMessages(prev => [...prev, { 
            id: Date.now(), 
            sender: "system", 
            text: "The AI is taking longer than expected to respond. Your message was sent and is still being processed. Complex requests (like code analysis) may take up to a few minutes.",
            isWarning: true
          }]);
        }, 40000); // 40 more seconds after the first message (total 60s)
      }, 20000); // First status after 20 seconds
      
    } else if (!socket || socket.readyState !== WebSocket.OPEN) {
      setMessages(prev => [...prev, { 
        id: Date.now(), 
        sender: "system", 
        text: "Cannot send message: Not connected to server. Trying to reconnect now...",
        isError: true
      }]);
      
      // Try to reconnect
      setConnectionAttempts(prev => prev + 1);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Generate a status indicator color based on connection status
  const getStatusColor = () => {
    switch (connectionStatus) {
      case "Connected": return "green";
      case "Connecting...": return "orange";
      case "Connection Error":
      case "Disconnected":
      case "Connection Failed": return "red";
      default: return "gray";
    }
  };

  return (
    <div className="chat-app">
      <header className="chat-header">
        <h1>MCP AI Chatbot</h1>
        <div className="connection-status">
          <span className="status-dot" style={{ backgroundColor: getStatusColor() }}></span>
          <span>{connectionStatus}</span>
          {connectionStatus !== "Connected" && (
            <button 
              className="reconnect-button"
              onClick={() => setConnectionAttempts(prev => prev + 1)}
            >
              Reconnect
            </button>
          )}
        </div>
      </header>
      
      <main className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h2>Welcome to MCP AI Chatbot!</h2>
            <p>Start chatting by typing a message below. I can help with:</p>
            <ul>
              <li>Answering questions about code and projects</li>
              <li>Searching and analyzing files</li>
              <li>Running tools and commands</li>
              <li>Understanding complex technical concepts</li>
            </ul>
          </div>
        )}
        
        {messages.map((msg) => (
          <div 
            key={`message-${msg.id}`} 
            className={`message ${msg.sender}-message 
            ${msg.isError ? 'error-message' : ''} 
            ${msg.isWarning ? 'warning-message' : ''} 
            ${msg.isStatus ? 'status-message' : ''}`}
          >
            {msg.sender === "system" ? (
              <div className={`system-message-content 
                ${msg.isError ? 'error' : ''} 
                ${msg.isWarning ? 'warning' : ''} 
                ${msg.isStatus ? 'status' : ''}`}>
                {msg.text}
              </div>
            ) : (
              <>
                <div className="message-avatar">
                  {msg.sender === "user" ? "You" : "AI"}
                </div>
                <div className="message-content">
                  {/* Format message content with line breaks with proper unique keys */}
                  {msg.text.split('\n').map((line, i) => (
                    <React.Fragment key={`${msg.id}-line-${i}`}>
                      {line}
                      {i < msg.text.split('\n').length - 1 && <br />}
                    </React.Fragment>
                  ))}
                </div>
              </>
            )}
          </div>
        ))}
        
        {isTyping && (
          <div className="message bot-message">
            <div className="message-avatar">AI</div>
            <div className="message-content typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </main>
      
      <footer className="input-container">
        <textarea
          ref={inputRef}
          className="message-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder={connectionStatus === "Connected" ? "Type your message here..." : "Waiting for connection..."}
          disabled={connectionStatus !== "Connected"}
          rows={1}
        />
        <button 
          className="send-button"
          onClick={sendMessage} 
          disabled={connectionStatus !== "Connected" || !input.trim()}
          aria-label="Send message"
        >
          <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </footer>
    </div>
  );
}

export default App;