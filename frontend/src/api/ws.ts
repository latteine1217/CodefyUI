type MessageHandler = (data: any) => void;

export class ExecutionWebSocket {
  private ws: WebSocket | null = null;
  private handlers: Map<string, MessageHandler[]> = new Map();

  connect(): Promise<void> {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/execution`);

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string);

        const typeHandlers = this.handlers.get(data.type) ?? [];
        typeHandlers.forEach((h) => h(data));

        const wildcardHandlers = this.handlers.get('*') ?? [];
        wildcardHandlers.forEach((h) => h(data));
      } catch {
        console.error('Failed to parse WebSocket message:', event.data);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    return new Promise<void>((resolve, reject) => {
      this.ws!.onopen = () => resolve();
      this.ws!.onerror = () => reject(new Error('WebSocket connection failed'));
    });
  }

  on(type: string, handler: MessageHandler): void {
    if (!this.handlers.has(type)) this.handlers.set(type, []);
    this.handlers.get(type)!.push(handler);
  }

  off(type: string, handler: MessageHandler): void {
    const handlers = this.handlers.get(type);
    if (handlers) {
      this.handlers.set(type, handlers.filter((fn) => fn !== handler));
    }
  }

  send(data: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not connected. Cannot send:', data);
    }
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const executionWs = new ExecutionWebSocket();
