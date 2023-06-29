import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ChatComponent } from './chat/chat.component'
import { ChatService } from './chat.service';
import { nl2brPipe } from './chat/nl2brPipe.filter';
@NgModule({
  imports: [CommonModule, FormsModule],
  declarations: [ChatComponent,nl2brPipe],
  providers: [ChatService],
})
export class AngularBotModule {}