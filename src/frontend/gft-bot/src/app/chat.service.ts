import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, throwError } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
export class Message {
  constructor(public author: string, public content: string) {}
}
export interface ChatMessage {
  role: {
    label: string;
  };
  content: string;
}
@Injectable()
export class ChatService {
  private apiUrl = 'http://localhost:8181/chat';

  constructor(private http: HttpClient) { }

  conversation = new BehaviorSubject<Message[]>([]);
  getBotAnswer(msg: string) {
    const userMessage = new Message('user', msg);
    this.conversation.next([userMessage]);
    this.getBotMessage(msg);
  }
  getBotMessage(question: string){
    const payload = {
      Question: question,
      DeploymentOrModelName: 'gpt-play'
    };
    this.http.post<ChatMessage>(this.apiUrl, payload).subscribe(
      result => {
        this.conversation.next(([new Message('bot', result.content)]));
      },
      (error: HttpErrorResponse) => {
        this.conversation.next(([new Message('bot', error.message)]));
      }
    );

  }
}