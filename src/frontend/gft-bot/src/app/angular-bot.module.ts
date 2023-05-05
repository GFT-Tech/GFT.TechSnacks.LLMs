import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ChatComponent } from './chat/chat.component'
import { ChatService } from './chat.service';
@NgModule({
  imports: [CommonModule, FormsModule],
  declarations: [ChatComponent],
  providers: [ChatService],
})
export class AngularBotModule {}